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
import os
import socket
import time
import warnings
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.pcie_ipc_policy import IpcLaunchConfig, IpcVariant
from flashinfer.comm.pcie_ipc_topology import PCIE_IPC_PROFILES

# (world_size, hidden, batch, blocks, threads, variant)
#
# Between them these cases select every kernel the header can dispatch to.
# Spelled out rather than resolved: the seed reaches five of the nine (world
# size, variant) pairs, and the other four -- (2, STAGED), (4, UNSTAGED),
# (8, STAGED), (8, FLAT_STAGED) -- are reachable only through an explicit
# config or the tuner.
_JOIN_TIMEOUT_S = 600

_CASES = [
    (2, 2048, 1, 32, 64, IpcVariant.UNSTAGED),  # tp2, unstaged
    (2, 2048, 8, 96, 64, IpcVariant.STAGED),  # tp2, staged
    (2, 2048, 128, 16, 128, IpcVariant.UNSTAGED),
    (4, 4096, 1, 1, 128, IpcVariant.STAGED),  # rsag push
    (4, 4096, 8, 1, 256, IpcVariant.STAGED_RING),  # rsag ring push
    (4, 4096, 128, 4, 256, IpcVariant.STAGED_RING),
    (4, 4096, 16, 8, 256, IpcVariant.UNSTAGED),  # one-shot push at 4 ranks
    (8, 6144, 1, 32, 128, IpcVariant.UNSTAGED),  # topo pack
    (8, 6144, 2, 1, 128, IpcVariant.FLAT_STAGED),  # generic rsag at 8 ranks
    (8, 6144, 8, 1, 256, IpcVariant.STAGED_RING),  # topo ring push
    (8, 6144, 128, 2, 256, IpcVariant.STAGED_RING),
    (8, 6144, 16, 8, 256, IpcVariant.STAGED),  # topo block (blocks % 4 == 0)
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
        # The profile only keys the tune cache, but the probe must still land
        # on a known one; which one is a property of the machine.
        assert ws.profile in PCIE_IPC_PROFILES, ws.profile_reason

        for _, hidden, batch, blocks, threads, variant in cases:
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

            # Admitted shapes must resolve and be correct, not merely the
            # right shape: a mis-selected kernel would otherwise go unnoticed.
            assert ws.supports(inp)
            torch.testing.assert_close(ws.all_reduce(inp), ref, rtol=0, atol=0)

            # The explicit config then pins the kernel this case exists for.
            out = ws.all_reduce(
                inp,
                config=IpcLaunchConfig(blocks, threads, variant),
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

        # Hidden size is not an admission criterion. Only the byte count is, and
        # this one fits.
        assert ws.supports(torch.empty(2, 4096, dtype=torch.bfloat16, device=device))
        # Fewer 16-byte packs than ranks: the reduce-scatter split would hand
        # the whole payload to one owner and leave the others idle.
        assert not ws.supports(torch.empty(8, dtype=torch.bfloat16, device=device))
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


# Batches straddling a seed block-count boundary at 4 ranks, hidden 4096: 32 ->
# one ring block, 64/70/80 -> two. Each pair grows the grid; the last two share
# one and are the control. Variant and thread count are constant, so the grid is
# the only thing that moves.
_GRID_CHANGE_BATCHES = [32, 64, 32, 70, 32, 80, 70]


def _grid_change_worker(world_size: int, rank: int, port: int) -> None:
    """Change the grid size between calls, which used to be an instant hang.

    The epoch used to be per block: each block flipped its own parity on exit,
    so the parity recorded how many times *that block* had run. Grow the grid
    and a block running for the first time reads the initial 0 and picks the
    half the previous call just used, with no slack at all.

    The sequence ends with ``80, 70`` deliberately. Those two calls have the
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
        # No profile pinned: the seed does not consult it, so the batch->blocks
        # mapping above holds on any machine.
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=hidden * max(_GRID_CHANGE_BATCHES),
            dtype=torch.bfloat16,
        )
        # Without a grid change there is no first-appearance block and nothing
        # under test, so the sequence is checked rather than assumed.
        grids = {
            ws.launch_config(
                torch.empty(b, hidden, dtype=torch.bfloat16, device=device)
            ).blocks
            for b in _GRID_CHANGE_BATCHES
        }
        assert len(grids) > 1, f"every batch resolves to the same grid: {grids}"
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
        one = IpcLaunchConfig(1, 256, IpcVariant.STAGED_RING)
        many = IpcLaunchConfig(4, 256, IpcVariant.STAGED_RING)
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


# One call per scratch region at 4 ranks: the staged kernels stage through
# kBlock, the one-shot push through kPack (see ScratchRegion in
# pcie_ipc_all_reduce.cuh). Paired with the batch each is issued at, so the
# region and the owner partition move together.
_REGION_ALTERNATION = [
    (96, IpcLaunchConfig(2, 256, IpcVariant.STAGED)),
    (1, IpcLaunchConfig(8, 256, IpcVariant.UNSTAGED)),
    (96, IpcLaunchConfig(2, 256, IpcVariant.STAGED)),
    (1, IpcLaunchConfig(8, 256, IpcVariant.UNSTAGED)),
    (1, IpcLaunchConfig(8, 256, IpcVariant.UNSTAGED)),
    (96, IpcLaunchConfig(2, 256, IpcVariant.STAGED)),
]


def _region_alternation_worker(world_size: int, rank: int, port: int) -> None:
    """Alternate the two scratch regions on one workspace.

    Each region carries its own call-level epoch counter, so only an
    interleaved sequence drives them against each other -- and only such a
    sequence catches a state pointer bound to the wrong region. A run that
    stays in one region never reads the other's leftovers and passes either way.

    The configurations are explicit because nothing has to choose them: the
    hazard is in the kernels, not in which kernel a shape resolves to.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        hidden = 4096
        # An edit that left every call in one region would still pass while
        # checking nothing.
        regions = {cfg.variant == IpcVariant.UNSTAGED for _, cfg in _REGION_ALTERNATION}
        assert regions == {True, False}, "the sequence stays in one scratch region"
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=hidden * max(b for b, _ in _REGION_ALTERNATION),
            dtype=torch.bfloat16,
        )
        payloads, refs = [], []
        for i, (b, _) in enumerate(_REGION_ALTERNATION):
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
        dist.barrier(group=group)

        for x, ref, (b, cfg) in zip(payloads, refs, _REGION_ALTERNATION, strict=True):
            torch.testing.assert_close(
                ws.all_reduce(x, config=cfg),
                ref,
                rtol=0,
                atol=0,
                msg=f"batch {b} with {cfg}",
            )
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [4])
def test_pcie_ipc_scratch_region_alternation(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _region_alternation_worker)


# Every TP8 variant at several launch configurations. The tuner can pick any of
# them for a neighbouring batch, so this is a sequence a caller can produce.
#
# The order is not arbitrary. Each call flips its region's epoch, so a sequence
# that alternates strictly between the sentinel and topology kernels locks each
# of them onto one half and the sentinel kernel never meets the other's
# leftovers -- it passes whatever region they are in. Two topology calls in a
# row dirty both halves, and two sentinel calls in a row then have to read both.
_MIXED_TP8 = [
    IpcLaunchConfig(32, 128, IpcVariant.UNSTAGED),
    IpcLaunchConfig(4, 128, IpcVariant.STAGED),
    IpcLaunchConfig(1, 128, IpcVariant.STAGED_RING),
    IpcLaunchConfig(1, 128, IpcVariant.FLAT_STAGED),
    IpcLaunchConfig(2, 256, IpcVariant.FLAT_STAGED),
    IpcLaunchConfig(12, 256, IpcVariant.UNSTAGED),
    IpcLaunchConfig(8, 256, IpcVariant.STAGED),
    IpcLaunchConfig(1, 512, IpcVariant.FLAT_STAGED),
]
# The owner partition is derived from the pack count, so the batch has to move
# too: a fixed shape only ever exercises one set of chunk boundaries.
_MIXED_TP8_BATCHES = [1, 2, 3, 5, 8]


def _mixed_variant_worker(world_size: int, rank: int, port: int) -> None:
    """Every TP8 variant interleaved on one workspace, across several shapes.

    Sentinel kernels and barrier kernels cannot share a scratch region (see
    ScratchRegion in pcie_ipc_all_reduce.cuh). A violation is wrong output on a
    subset of ranks, not a hang, which is why this asserts group-wide.

    Distinct payloads per call are essential. With one payload reused, a stale
    read returns the value the call would have computed anyway and the whole
    sequence passes while proving nothing.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        hidden = 6144
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=hidden * max(_MIXED_TP8_BATCHES),
            dtype=torch.bfloat16,
            profile="rootcplx",
        )
        call = 0
        for batch in _MIXED_TP8_BATCHES:
            for config in _MIXED_TP8:
                inp = torch.randint(
                    0, 16, (batch, hidden), dtype=torch.int32, device=device
                ).to(torch.bfloat16) + float(call % 11)
                ref = inp.clone()
                dist.all_reduce(ref, group=group)
                out = ws.all_reduce(inp, config=config)
                # Group-wide: the hazard corrupts a subset of ranks, so a
                # rank-local assertion can pass on rank 0 while the collective
                # is wrong elsewhere.
                wrong = torch.tensor([int((out != ref).sum().item())], device=device)
                dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
                assert int(wrong.item()) == 0, (
                    f"call {call} (batch {batch}, {config}) produced "
                    f"{int(wrong.item())} wrong elements: a variant is reading "
                    "scratch another variant left dirty"
                )
                call += 1
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [8])
def test_pcie_ipc_mixed_variants_share_no_scratch(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _mixed_variant_worker)


# Legal (blocks, threads, variant) per world size, one entry per dispatchable
# kernel. blocks is the smallest each kernel accepts: the TP8 block kernel
# derives its chunk from blockIdx.x & 3 and strides by gridDim.x >> 2.
_TINY_CONFIGS = {
    2: [
        IpcLaunchConfig(1, 64, IpcVariant.UNSTAGED),
        IpcLaunchConfig(1, 64, IpcVariant.STAGED),
    ],
    4: [
        IpcLaunchConfig(1, 64, IpcVariant.UNSTAGED),
        IpcLaunchConfig(1, 64, IpcVariant.STAGED),
        IpcLaunchConfig(1, 64, IpcVariant.STAGED_RING),
    ],
    8: [
        IpcLaunchConfig(1, 64, IpcVariant.UNSTAGED),
        IpcLaunchConfig(4, 64, IpcVariant.STAGED),
        IpcLaunchConfig(1, 64, IpcVariant.STAGED_RING),
        IpcLaunchConfig(1, 64, IpcVariant.FLAT_STAGED),
    ],
}


def _tiny_payload_worker(world_size: int, rank: int, port: int) -> None:
    """Payloads with fewer 16-byte packs than ranks, on every variant.

    The reduce-scatter split gives each rank ``num_packs // world_size`` packs,
    which is zero once the payload is smaller than one pack per rank. The
    staged kernels have to agree on who owns the payload in that case; a kernel
    that writes to one owner and polls another spins forever rather than
    returning a wrong answer, so the bounded join in multi_process_parallel is
    what turns a regression into a report.

    Reached through explicit configurations: admission requires at least one
    pack per rank, so no shape the policy accepts lands here.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        pack_elems = 8  # 16 bytes / bfloat16
        numels = [pack_elems * n for n in range(1, world_size + 1)]
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=max(numels),
            dtype=torch.bfloat16,
            profile="rootcplx",
        )
        for numel in numels:
            for config in _TINY_CONFIGS[world_size]:
                inp = torch.randint(
                    0, 16, (1, numel), dtype=torch.int32, device=device
                ).to(torch.bfloat16)
                ref = inp.clone()
                dist.all_reduce(ref, group=group)
                out = ws.all_reduce(inp, config=config)
                wrong = torch.tensor([int((out != ref).sum().item())], device=device)
                dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
                assert int(wrong.item()) == 0, (
                    f"numel {numel} ({numel // pack_elems} packs, {world_size} "
                    f"ranks) with {config} produced {int(wrong.item())} wrong "
                    "elements"
                )
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_tiny_payload_every_variant(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _tiny_payload_worker)


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


_TUNE_HIDDEN = 6144
_TUNE_BATCHES = (1, 2, 4)


def _tune_cache_path(tmpdir: str) -> str:
    return os.path.join(tmpdir, "pcie_ipc_tune.json")


def _tune_worker(world_size: int, rank: int, port: int, tmpdir: str) -> None:
    """Tune, then check the group agrees, the answer is right, and it persists.

    Autotuning a collective is only safe if every rank ends up on the same
    kernel. Nothing in the kernels checks that -- they spin without a timeout --
    so the agreement is asserted here rather than assumed.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        path = _tune_cache_path(tmpdir)
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=_TUNE_HIDDEN * max(_TUNE_BATCHES),
            dtype=torch.bfloat16,
            profile="rootcplx",
            tune_batches=_TUNE_BATCHES,
        )
        before = {b: ws.supports(_tune_input(b, device)) for b in _TUNE_BATCHES}
        ws.tune([_TUNE_HIDDEN], cache=path, warmup=2, repeat=5)

        resolved = {}
        for batch in _TUNE_BATCHES:
            inp = _tune_input(batch, device)
            # Tuning must not change which shapes are claimed: admission needs a
            # comparison against the caller's fallback, and that fallback is not
            # one of the candidates.
            assert ws.supports(inp) == before[batch], batch
            config = ws.tuned_launch_config(inp)
            assert config is not None
            resolved[batch] = repr(config)

            ref = inp.clone()
            dist.all_reduce(ref, group=group)
            wrong = torch.tensor(
                [int((ws.all_reduce(inp) != ref).sum().item())], device=device
            )
            dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
            assert int(wrong.item()) == 0, (
                f"batch {batch} is wrong after tuning with {config}"
            )

        gathered = [None] * world_size
        dist.all_gather_object(gathered, resolved, group=group)
        assert all(g == gathered[0] for g in gathered), (
            f"ranks resolved different configurations: {gathered}"
        )
        if rank == 0:
            assert os.path.isfile(path), "rank 0 must persist the tuned cache"
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


def _tune_input(batch: int, device: torch.device) -> torch.Tensor:
    return torch.randint(
        0, 16, (batch, _TUNE_HIDDEN), dtype=torch.int32, device=device
    ).to(torch.bfloat16)


def _tune_reuse_worker(world_size: int, rank: int, port: int, tmpdir: str) -> None:
    """A second process must reuse the persisted cache without re-tuning.

    Loading has to be eager and identical on every rank. A rank that picked up
    the file later than its peers -- or dropped entries the others kept --
    would run a different kernel, and the group hangs rather than errors.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().clear_cache()
        AutoTuner.get().load_configs(_tune_cache_path(tmpdir))
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=_TUNE_HIDDEN * max(_TUNE_BATCHES),
            dtype=torch.bfloat16,
            profile="rootcplx",
            tune_batches=_TUNE_BATCHES,
        )
        assert not AutoTuner.get().is_tuning_mode, (
            "reuse must not enter tuning mode; a collective sweep here would be "
            "a several-second stall in a serving process"
        )
        resolved, differs = {}, 0
        for batch in _TUNE_BATCHES:
            inp = _tune_input(batch, device)
            config = ws.tuned_launch_config(inp)
            resolved[batch] = repr(config)
            if config != ws.launch_config(inp):
                differs += 1
            ref = inp.clone()
            dist.all_reduce(ref, group=group)
            wrong = torch.tensor(
                [int((ws.all_reduce(inp) != ref).sum().item())], device=device
            )
            dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
            assert int(wrong.item()) == 0, batch
        gathered = [None] * world_size
        dist.all_gather_object(gathered, resolved, group=group)
        assert all(g == gathered[0] for g in gathered), gathered
        # If the reloaded cache never disagreed with the seed, the test would
        # pass while proving nothing about persistence.
        assert differs > 0, (
            "the persisted cache resolved to the seed configuration everywhere, "
            "so this test did not exercise the cache"
        )
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


def _tune_cardinality_worker(
    world_size: int, rank: int, port: int, tmpdir: str
) -> None:
    """Every rank must issue the identical number of launches while tuning.

    The whole search is a chain of unsynchronised collectives. One rank issuing
    a different number of them -- because it screened a different candidate set,
    or returned early -- leaves the rest spinning with no timeout. Counting is
    the only way to observe it from outside.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=_TUNE_HIDDEN * 2,
            dtype=torch.bfloat16,
            profile="rootcplx",
            tune_batches=(1, 2),
        )
        launches = 0
        original = ws._launch

        def counting(*args, **kwargs):
            nonlocal launches
            launches += 1
            return original(*args, **kwargs)

        ws._launch = counting
        try:
            ws.tune([_TUNE_HIDDEN], cache=_tune_cache_path(tmpdir), warmup=2, repeat=5)
        finally:
            ws._launch = original

        counts = [None] * world_size
        dist.all_gather_object(counts, launches, group=group)
        assert len(set(counts)) == 1, (
            f"ranks issued different numbers of launches while tuning: {counts}"
        )
        assert counts[0] > 0
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


def _tune_gate_worker(world_size: int, rank: int, port: int, tmpdir: str) -> None:
    """A candidate that computes the wrong answer must not survive screening.

    The autotuner ranks purely on time, and this protocol's failure mode is a
    sentinel poll returning stale data -- wrong *and* fast. The corruption is
    injected in Python rather than by rebuilding a broken kernel: the point is
    to test the gate, and a genuinely broken build can spin instead of finish.
    """
    ws = None
    group = None
    tune_group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        from flashinfer.autotuner import set_autotune_process_group

        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=_TUNE_HIDDEN * 2,
            dtype=torch.bfloat16,
            profile="rootcplx",
            tune_batches=(1, 2),
        )
        # Screening only runs when a real search is safe, so install the group
        # the search would have.
        tune_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        set_autotune_process_group(tune_group)
        from flashinfer.comm.pcie_ipc_tuning import candidate_tactics, tactic_to_config

        poisoned = tactic_to_config(candidate_tactics(world_size, ws.max_blocks)[3])
        original = ws._launch

        def sabotage(inp, out, config, enable_pdl=False):
            original(inp, out, config, enable_pdl)
            if config == poisoned:
                out.add_(1)

        ws._launch = sabotage
        try:
            inp = _tune_input(2, device)
            survivors = ws._runner.get_valid_tactics([inp], None)
        finally:
            ws._launch = original
            set_autotune_process_group(None)

        from flashinfer.comm.pcie_ipc_tuning import config_to_tactic

        assert config_to_tactic(poisoned) not in survivors, (
            f"{poisoned} computed the wrong answer but survived screening"
        )
        assert len(survivors) > 1, "screening must not reject everything"
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if tune_group is not None:
            dist.destroy_process_group(tune_group)
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [8])
def test_pcie_ipc_tune_cardinality(world_size: int, tmp_path) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(
        world_size, _tune_cardinality_worker, (str(tmp_path),), timeout_s=900
    )


@pytest.mark.parametrize("world_size", [8])
def test_pcie_ipc_tune_gate_excludes_wrong_candidates(
    world_size: int, tmp_path
) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(
        world_size, _tune_gate_worker, (str(tmp_path),), timeout_s=600
    )


def _untuned_warning_worker(world_size: int, rank: int, port: int, tmpdir: str) -> None:
    """The untuned warning fires once, and only when the machine is untuned.

    Three ways to get this wrong, all user-visible. Warning per call turns a
    serving loop into a log flood. Warning while ``autotune(True)`` is open
    advises the caller to tune in the middle of tuning -- that path skips the
    hot cache, so every call reaches the cold path. And warning after
    :meth:`tune` tells them to redo work they just did.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        from flashinfer.autotuner import AutoTuner, autotune

        AutoTuner.get().clear_cache()
        path = os.path.join(tmpdir, f"absent_ws{world_size}.json")
        assert not os.path.exists(path)

        def _build():
            return comm.PcieIpcAllReduceWorkspace(
                group=group,
                max_numel=_TUNE_HIDDEN * max(_TUNE_BATCHES),
                dtype=torch.bfloat16,
                profile="rootcplx",
                tune_batches=_TUNE_BATCHES,
                tune_cache=path,
            )

        def _untuned_warnings(fn):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fn()
            return [w for w in caught if "seed launch configurations" in str(w.message)]

        ws = _build()
        first = _untuned_warnings(lambda: ws.all_reduce(_tune_input(1, device)))
        assert len(first) == 1, first
        # A different shape, so this is another cold-path call rather than a hot
        # cache hit: the flag has to be what stops it, not the shape cache.
        again = _untuned_warnings(lambda: ws.all_reduce(_tune_input(2, device)))
        assert not again, again

        # A fresh workspace on the same untuned machine must not warn while it
        # is the one doing the tuning.
        ws.destroy()
        ws = _build()

        def _tune_in_process():
            with autotune(True, tuning_buckets=_TUNE_BATCHES, round_up=False):
                for batch in _TUNE_BATCHES:
                    ws.tuned_launch_config(_tune_input(batch, device))

        during = _untuned_warnings(_tune_in_process)
        assert not during, during

        # And silent afterwards, including on a shape tuning did not cover.
        ws.tune([_TUNE_HIDDEN], cache=path, warmup=2, repeat=5)
        after = _untuned_warnings(lambda: ws.all_reduce(_tune_input(4, device)))
        assert not after, after
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [8])
def test_pcie_ipc_untuned_warning(world_size: int, tmp_path) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _untuned_warning_worker, (str(tmp_path),))


@pytest.mark.parametrize("world_size", [8])
def test_pcie_ipc_tune_end_to_end(world_size: int, tmp_path) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    tmpdir = str(tmp_path)
    multi_process_parallel(world_size, _tune_worker, (tmpdir,), timeout_s=1200)
    multi_process_parallel(world_size, _tune_reuse_worker, (tmpdir,), timeout_s=600)


def _standard_idiom_worker(world_size: int, rank: int, port: int, tmpdir: str) -> None:
    """The library's usual ``with autotune(True)`` idiom must work here too.

    ``tune()`` is a convenience, not the only entry point -- a caller who knows
    FlashInfer should not have to learn a second way to tune one operator.
    """
    ws = None
    group = None
    tune_group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        from flashinfer.autotuner import (
            AutoTuner,
            autotune,
            set_autotune_process_group,
        )

        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=_TUNE_HIDDEN * 2,
            dtype=torch.bfloat16,
            profile="rootcplx",
            tune_batches=(1, 2),
        )
        path = os.path.join(tmpdir, "idiom.json")
        tune_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        set_autotune_process_group(tune_group)
        try:
            with autotune(True, cache=path):
                for _ in range(2):
                    for batch in (1, 2):
                        ws.all_reduce(_tune_input(batch, device))
        finally:
            set_autotune_process_group(None)

        resolved = {}
        for batch in (1, 2):
            inp = _tune_input(batch, device)
            ref = inp.clone()
            dist.all_reduce(ref, group=group)
            wrong = torch.tensor(
                [int((ws.all_reduce(inp) != ref).sum().item())], device=device
            )
            dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
            assert int(wrong.item()) == 0, batch
            resolved[batch] = repr(ws.tuned_launch_config(inp))
        gathered = [None] * world_size
        dist.all_gather_object(gathered, resolved, group=group)
        assert all(g == gathered[0] for g in gathered), gathered
        assert AutoTuner.get().stats.tuned_op_successful_configs.get(
            comm.PCIE_IPC_CUSTOM_OP
        ), "the standard idiom must actually have profiled something"
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if tune_group is not None:
            dist.destroy_process_group(tune_group)
        if group is not None:
            dist.destroy_process_group(group)


def _stray_tuning_worker(world_size: int, rank: int, port: int, tmpdir: str) -> None:
    """A tuning session that forgot the reduction group must degrade, not hang.

    Tuning mode is process-global, so a caller tuning its GEMMs sweeps this
    operator too. Without a reduction over the candidate timings every rank
    would argmin independently and they would not agree on a kernel -- which
    this protocol does not survive. The search has to notice and decline.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        from flashinfer.autotuner import autotune, get_autotune_process_group

        assert get_autotune_process_group() is None
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=_TUNE_HIDDEN * 2,
            dtype=torch.bfloat16,
            profile="rootcplx",
            tune_batches=(1, 2),
        )
        with pytest.warns(RuntimeWarning, match="no matching"), autotune(True):
            for batch in (1, 2):
                inp = _tune_input(batch, device)
                ref = inp.clone()
                dist.all_reduce(ref, group=group)
                wrong = torch.tensor(
                    [int((ws.all_reduce(inp) != ref).sum().item())], device=device
                )
                dist.all_reduce(wrong, op=dist.ReduceOp.MAX, group=group)
                assert int(wrong.item()) == 0, batch
                # The seed's answer, unchanged: nothing was searched.
                assert ws.tuned_launch_config(inp) == ws.launch_config(inp)
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [8])
def test_pcie_ipc_tune_via_standard_idiom(world_size: int, tmp_path) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(
        world_size, _standard_idiom_worker, (str(tmp_path),), timeout_s=900
    )


@pytest.mark.parametrize("world_size", [8])
def test_pcie_ipc_stray_tuning_session_degrades(world_size: int, tmp_path) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(
        world_size, _stray_tuning_worker, (str(tmp_path),), timeout_s=300
    )
