"""Copy-engine ring: the protocol properties the SM kernels do not have.

The ring synchronises through monotonic flags rather than through the payload,
and it stages into scratch guarded by an end-of-call handshake rather than by an
epoch double buffer. Both are the reason it can be a second data plane at all,
and neither is exercised by the SM path's tests.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.distributed as dist

from flashinfer import comm
from flashinfer.comm.pcie_ipc_policy import IpcLaunchConfig, IpcVariant
from tests.comm.test_pcie_ipc_all_reduce import (
    _init_process_group,
    multi_process_parallel,
)

_HIDDEN = 6144
_BATCH = 64

# Both schedules, because they do not share a protocol. The island variant is
# filtered out of the tuner's candidates on any fabric its 4+4 grouping does not
# describe, so on such a box these tests are the only thing that runs it at all
# -- an explicit `config=` is documented to reach a kernel the tuner would not
# choose. Reaching it matters here: it stages into two peers' scratch rather
# than one, which is the part the end-of-call handshake has to cover.
_VARIANTS = [IpcVariant.COPY_ENGINE_RING, IpcVariant.COPY_ENGINE_ISLAND]


def _ce_config(
    pieces: int = 2, variant: IpcVariant = IpcVariant.COPY_ENGINE_RING
) -> IpcLaunchConfig:
    return IpcLaunchConfig(blocks=pieces, threads=256, variant=variant)


def _skip_unless_runnable(world_size: int, variant: IpcVariant) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip(f"not enough GPUs: need {world_size}")
    if variant == IpcVariant.COPY_ENGINE_ISLAND and world_size != 8:
        pytest.skip("the island schedule is a 4+4 decomposition, world_size 8 only")


def _graph_replay_worker(
    world_size: int, rank: int, port: int, variant: IpcVariant
) -> None:
    _init_process_group(world_size, rank, port)
    device = torch.device("cuda", rank)
    group = dist.group.WORLD
    ws: Optional[comm.PcieIpcAllReduceWorkspace] = None
    try:
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=_BATCH * _HIDDEN, dtype=torch.bfloat16
        )
        config = _ce_config(variant=variant)
        inp = torch.randint(
            0, 16, (_BATCH, _HIDDEN), dtype=torch.int32, device=device
        ).to(torch.bfloat16)
        out = torch.empty_like(inp)
        ref = inp.clone()
        dist.all_reduce(ref, group=group)

        # Warm up outside the graph: the first launch of a specialisation loads
        # its module, which is illegal inside a capture.
        for _ in range(3):
            ws.all_reduce(inp, out=out, config=config)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

        for captured in (3, 4):
            ws.rebind_stream()  # capture warms up on a side stream
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(captured):
                    ws.all_reduce(inp, out=out, config=config)
            # Eager calls are interleaved with the replays because a
            # replay-only loop would also pass if the flag counters were reset
            # on every call; only calls that advance the counters outside the
            # graph tell the two apart.
            for i in range(8):
                graph.replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(out, ref, rtol=0, atol=0)
                if i % 2 == 0:
                    ws.rebind_stream()
                    ws.all_reduce(inp, out=out, config=config)
                    torch.cuda.synchronize(device)
                    torch.testing.assert_close(out, ref, rtol=0, atol=0)
            dist.barrier(group=group)
            del graph
    finally:
        if ws is not None:
            ws.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def _skewed_ranks_worker(
    world_size: int, rank: int, port: int, variant: IpcVariant, dtype: torch.dtype
) -> None:
    """Back-to-back calls with one rank deliberately behind.

    Despite the name this does NOT cover the end-of-call rendezvous: deleting
    either handshake leaves it passing 20/20 on both schedules, while truncating
    the island all-gather by one step fails it immediately -- so the detector
    works and header edits do reach the compiled binary, which is what rules out
    a stale JIT cache as the explanation for the 20/20. It simply cannot open
    the window the rendezvous closes, because the ring's own data dependencies
    (`r@k` implies `(r-1)@(k-1)`) already serialise it. If the rendezvous is ever
    changed or removed, this test will not notice; the scenario that would is not
    currently constructible from the public API, see the comment at the handshake.

    The skew is device-side (torch.cuda._sleep) because the collective is
    stream-ordered and a host delay would only delay submission. The laggard
    rotates so that ranks 3 and 7 take a turn: they sit at the island wrap, and
    they are the ranks a single global handshake slot would leave covering
    neither peer they stage into -- which is why there are two slots. See
    ce_flag_slots in pcie_ipc_all_reduce.cuh.
    """
    _init_process_group(world_size, rank, port)
    device = torch.device("cuda", rank)
    group = dist.group.WORLD
    ws: Optional[comm.PcieIpcAllReduceWorkspace] = None
    try:
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=_BATCH * _HIDDEN, dtype=dtype
        )
        config = _ce_config(variant=variant)
        out = torch.empty(_BATCH, _HIDDEN, dtype=dtype, device=device)
        for i in range(8 * world_size):
            # An iteration-dependent value, so staging left over from call i-1
            # cannot be mistaken for a correct result for call i.
            inp = torch.full(
                (_BATCH, _HIDDEN), float(i % 16), dtype=dtype, device=device
            )
            if rank == i % world_size:
                torch.cuda._sleep(2_000_000)
            ws.all_reduce(inp, out=out, config=config)
            torch.cuda.synchronize(device)
            expected = torch.full_like(out, float(i % 16) * world_size)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda v: v.name.lower())
@pytest.mark.parametrize("world_size", [4, 8])
def test_ce_ring_survives_graph_replay(world_size: int, variant: IpcVariant) -> None:
    _skip_unless_runnable(world_size, variant)
    multi_process_parallel(world_size, _graph_replay_worker, args=(variant,))


# Both 2-byte dtypes take the same path apart from the packed_add_u4
# specialisation, which the ring shares with the SM kernels, so fp16 rides on
# the correctness test alone; the replay test stays bf16 because the flag
# kernels it exercises are not templated on dtype at all.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda v: v.name.lower())
@pytest.mark.parametrize("world_size", [4, 8])
def test_ce_ring_is_correct_with_skewed_ranks(
    world_size: int, variant: IpcVariant, dtype: torch.dtype
) -> None:
    _skip_unless_runnable(world_size, variant)
    multi_process_parallel(world_size, _skewed_ranks_worker, args=(variant, dtype))
