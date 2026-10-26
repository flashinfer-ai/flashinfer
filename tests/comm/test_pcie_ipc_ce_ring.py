"""Copy-engine ring: the protocol properties the SM kernels do not have.

The ring synchronises through monotonic flags rather than through the payload,
and it stages into scratch guarded by an end-of-call handshake rather than by an
epoch double buffer. Both are the reason it can be a second data plane at all,
and neither is exercised by the SM path's tests. The pipelined SM120 ring has
no handshake at all -- its own step dependencies cover reuse, see
ce_ring_all_reduce_piped -- and sizes its chunks from the ring's slow hops, so
it is checked under every hop mask, depth and uneven split as well.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
import torch.distributed as dist

from flashinfer import comm
from flashinfer.comm.pcie_ipc_ar import get_pcie_ipc_comm_module
from flashinfer.comm.pcie_ipc_policy import IpcLaunchConfig, IpcVariant
from tests.comm.test_pcie_ipc_all_reduce import (
    _init_process_group,
    multi_process_parallel,
)

pytestmark = pytest.mark.node_exclusive

_HIDDEN = 6144
_BATCH = 64

# Both schedules, because they do not share a protocol. The island variant is
# filtered out of the tuner's candidates on any fabric its 4+4 grouping does not
# describe, so on such a box these tests are the only thing that runs it at all
# -- an explicit `config=` is documented to reach a kernel the tuner would not
# choose. Reaching it matters here: it stages into two peers' scratch rather
# than one, which is the part the end-of-call handshake has to cover.
_VARIANTS = [
    IpcVariant.COPY_ENGINE_RING,
    IpcVariant.COPY_ENGINE_ISLAND,
    IpcVariant.COPY_ENGINE_RING_MEMOP,
    IpcVariant.COPY_ENGINE_RING_PIPED,
]

# The stream-publication variants, SM120 only.
_MEMOP_VARIANTS = (IpcVariant.COPY_ENGINE_RING_MEMOP, IpcVariant.COPY_ENGINE_RING_PIPED)


def _ce_config(
    pieces: int = 2, variant: IpcVariant = IpcVariant.COPY_ENGINE_RING
) -> IpcLaunchConfig:
    return IpcLaunchConfig(blocks=pieces, threads=256, variant=variant)


def _default_depth(variant: IpcVariant) -> int:
    # The memop ring is single-piece; depth 3 makes the pipelined ring split
    # unevenly wherever the probe finds a slow hop.
    if variant == IpcVariant.COPY_ENGINE_RING_MEMOP:
        return 1
    return 3 if variant == IpcVariant.COPY_ENGINE_RING_PIPED else 2


def _skip_unless_runnable(world_size: int, variant: IpcVariant) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip(f"not enough GPUs: need {world_size}")
    if variant == IpcVariant.COPY_ENGINE_ISLAND and world_size != 8:
        pytest.skip("the island schedule is a 4+4 decomposition, world_size 8 only")
    if variant in _MEMOP_VARIANTS:
        if any(
            torch.cuda.get_device_capability(rank) != (12, 0)
            for rank in range(world_size)
        ):
            pytest.skip("the memop protocol requires SM120 on every participating GPU")
        module = get_pcie_ipc_comm_module()
        for rank in range(world_size):
            with torch.cuda.device(rank):
                assert module.memop_supported(), (
                    "SM120 must admit the default-supported v2 32-bit stream writes"
                )


def _graph_replay_worker(
    world_size: int, rank: int, port: int, variant: IpcVariant, dtype: torch.dtype
) -> None:
    _init_process_group(world_size, rank, port)
    device = torch.device("cuda", rank)
    group = dist.group.WORLD
    ws: Optional[comm.PcieIpcAllReduceWorkspace] = None
    try:
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=_BATCH * _HIDDEN, dtype=dtype
        )
        if variant in _MEMOP_VARIANTS:
            assert ws.memop_supported, "the test must exercise the memop protocol"
        config = _ce_config(_default_depth(variant), variant)
        inp = torch.randint(
            0, 16, (_BATCH, _HIDDEN), dtype=torch.int32, device=device
        ).to(dtype)
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
                inp.fill_(float((rank + i) % 16))
                ref.fill_(float(sum((peer + i) % 16 for peer in range(world_size))))
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
    The pipelined ring has no rendezvous to delete: these dependencies are the
    whole of its reuse argument, and this is the test that exercises them.

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
        if variant in _MEMOP_VARIANTS:
            assert ws.memop_supported, "the test must exercise the memop protocol"
        config = _ce_config(_default_depth(variant), variant)
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda v: v.name.lower())
@pytest.mark.parametrize("world_size", [4, 8])
def test_ce_ring_survives_graph_replay(
    world_size: int, variant: IpcVariant, dtype: torch.dtype
) -> None:
    _skip_unless_runnable(world_size, variant)
    multi_process_parallel(world_size, _graph_replay_worker, args=(variant, dtype))


# Both 2-byte dtypes share the flag protocol and specialize packed_add_u4.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("variant", _VARIANTS, ids=lambda v: v.name.lower())
@pytest.mark.parametrize("world_size", [4, 8])
def test_ce_ring_is_correct_with_skewed_ranks(
    world_size: int, variant: IpcVariant, dtype: torch.dtype
) -> None:
    _skip_unless_runnable(world_size, variant)
    multi_process_parallel(world_size, _skewed_ranks_worker, args=(variant, dtype))


def _mixed_memop_worker(world_size, rank, port, dtype):
    _init_process_group(world_size, rank, port)
    device = torch.device("cuda", rank)
    ws = None
    try:
        # At TP8 each one-MiB shard admits two half-MiB pieces.
        numel = 8 * 1024 * 1024 // 2
        ws = comm.PcieIpcAllReduceWorkspace(dist.group.WORLD, numel, dtype=dtype)
        assert ws.memop_supported, "the mixed test must exercise the memop protocol"
        inp = torch.empty(numel, dtype=dtype, device=device)
        out = torch.empty_like(inp)
        # The pipelined ring keeps no handshake and its own region, so it sits
        # between schedules that do, at depths with different piece counts.
        configs = [
            _ce_config(1, IpcVariant.COPY_ENGINE_RING_MEMOP),
            _ce_config(3, IpcVariant.COPY_ENGINE_RING_PIPED),
            _ce_config(2, IpcVariant.COPY_ENGINE_RING),
            _ce_config(1, IpcVariant.COPY_ENGINE_RING_PIPED),
            _ce_config(2, IpcVariant.COPY_ENGINE_RING_MEMOP),
            _ce_config(4, IpcVariant.COPY_ENGINE_RING_PIPED),
        ]
        if world_size == 8:
            configs.append(_ce_config(1, IpcVariant.COPY_ENGINE_ISLAND))
            configs.append(_ce_config(2, IpcVariant.COPY_ENGINE_RING_PIPED))
        configs.append(_ce_config(1, IpcVariant.COPY_ENGINE_RING_MEMOP))
        for iteration in range(3):
            for step, config in enumerate(configs):
                value = (iteration * len(configs) + step + rank) % 16
                inp.fill_(float(value))
                if rank == step % world_size:
                    torch.cuda._sleep(2_000_000)
                ws.all_reduce(inp, out=out, config=config)
                torch.cuda.synchronize(device)
                expected = sum(
                    (iteration * len(configs) + step + peer) % 16
                    for peer in range(world_size)
                )
                torch.testing.assert_close(
                    out, torch.full_like(out, expected), rtol=0, atol=0
                )
        dist.barrier()
    finally:
        if ws is not None:
            ws.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("world_size", [4, 8])
def test_ce_memop_preserves_mixed_protocol_reuse(world_size, dtype):
    _skip_unless_runnable(world_size, IpcVariant.COPY_ENGINE_RING_MEMOP)
    multi_process_parallel(world_size, _mixed_memop_worker, args=(dtype,))


def _piped_geometry_worker(world_size, rank, port, dtype, slow_hops):
    """Exact sums under one hop mask, every depth, and splits that do not divide.

    The geometry decides which elements travel in which piece and slot, so an
    off-by-one in a boundary drops or doubles a pack -- exactly what random
    small integers, compared at zero tolerance, catch. The mask also changes
    which slots the no-handshake reuse argument has to cover, so the calls go
    back to back with no collective between them, one rank skewed per call and
    the depth changing every call; a replayed graph rides along.
    """
    _init_process_group(world_size, rank, port)
    device = torch.device("cuda", rank)
    group = dist.group.WORLD
    ws = None
    try:
        pack = 8
        # Whole shards of 37 and 1001 packs per rank, then the full capacity.
        numels = [world_size * pack * 37, world_size * pack * 1001, _BATCH * _HIDDEN]
        ws = comm.PcieIpcAllReduceWorkspace(
            group, max(numels), dtype=dtype, ce_slow_hops=slow_hops
        )
        assert ws.memop_supported, "the test must exercise the pipelined ring"
        if slow_hops is not None:
            assert ws.slow_hops == slow_hops
        gen = torch.Generator()

        def draw(call, peer, numel):
            # On the host, so every rank can rebuild every peer's input.
            gen.manual_seed(1000 * call + peer)
            return torch.randint(0, 16, (numel,), generator=gen, dtype=torch.int32)

        depths = range(1, 5)
        call = 0
        for numel in numels:
            # Staged before the first launch: a pageable copy in between would
            # synchronise this rank's stream and space the calls out.
            inps = [
                draw(call + i, rank, numel).to(device=device, dtype=dtype)
                for i in range(len(depths))
            ]
            outs = [torch.full_like(inp, float("nan")) for inp in inps]
            torch.cuda.synchronize(device)
            # Nothing but the ring keeps a rank from running ahead into the next
            # call, at another depth and so with other pieces in flight.
            for i, depth in enumerate(depths):
                if rank == (call + i) % world_size:
                    torch.cuda._sleep(2_000_000)
                config = _ce_config(depth, IpcVariant.COPY_ENGINE_RING_PIPED)
                ws.all_reduce(inps[i], out=outs[i], config=config)
            torch.cuda.synchronize(device)
            for i, out in enumerate(outs):
                ref = sum(draw(call + i, peer, numel) for peer in range(world_size))
                torch.testing.assert_close(
                    out, ref.to(device=device, dtype=dtype), rtol=0, atol=0
                )
            call += len(depths)

        config = _ce_config(3, IpcVariant.COPY_ENGINE_RING_PIPED)
        inp = torch.empty(numels[1], dtype=dtype, device=device)
        out = torch.empty_like(inp)
        inp.fill_(1.0)
        ws.all_reduce(inp, out=out, config=config)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        ws.rebind_stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(3):
                ws.all_reduce(inp, out=out, config=config)
        for i in range(4):
            inp.fill_(float((rank + 3 * i) % 16))
            graph.replay()
            torch.cuda.synchronize(device)
            expected = float(sum((peer + 3 * i) % 16 for peer in range(world_size)))
            torch.testing.assert_close(
                out, torch.full_like(out, expected), rtol=0, atol=0
            )
        dist.barrier(group=group)
        del graph
    finally:
        if ws is not None:
            ws.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


# Probed, an even split, one slow hop (two enlarged chunks; at 8 ranks their
# weight is capped), both socket crossings of a two-socket machine whatever the
# probe finds here, and every hop slow (every chunk skipped by one: an even
# split again).
_HOP_MASKS = {
    "probed": lambda world_size: None,
    "even": lambda world_size: 0,
    "one": lambda world_size: 1 << 1,
    "two_socket": lambda world_size: (
        (1 << (world_size // 2 - 1)) | (1 << (world_size - 1))
    ),
    "all": lambda world_size: (1 << world_size) - 1,
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("mask", sorted(_HOP_MASKS))
@pytest.mark.parametrize("world_size", [4, 8])
def test_ce_piped_ring_is_exact_under_every_hop_mask(world_size, mask, dtype):
    _skip_unless_runnable(world_size, IpcVariant.COPY_ENGINE_RING_PIPED)
    slow_hops = _HOP_MASKS[mask](world_size)
    multi_process_parallel(world_size, _piped_geometry_worker, args=(dtype, slow_hops))
