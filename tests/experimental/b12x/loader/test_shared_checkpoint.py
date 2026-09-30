"""Qualify shared checkpoint completion across four independent CUDA workers."""

from datetime import timedelta
import json
import os
from pathlib import Path
from threading import Event, Thread
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@pytest.mark.parametrize("fails", [False, True])
def test_reader_initialization_overlaps_routing_and_close_joins(fails):
    from b12x.loader._shared_checkpoint import SharedReadGroup

    entered, release, closed = Event(), Event(), Event()
    executor = object()

    def create(*args):
        entered.set()
        assert release.wait(5)
        if fails:
            raise RuntimeError("reader initialization failed")
        return executor

    native = SimpleNamespace(owner_create=create, owner_close=Mock())
    group = object.__new__(SharedReadGroup)
    group.native = group.executor = group._initializer = group._initialization = None
    group.unsafe, group.device, group.files = False, 0, {}
    session = SimpleNamespace(_gds=native, io_threads=8,
                              _copy_programs=[SimpleNamespace(function=1), SimpleNamespace(function=2)])
    group.start(session)
    assert entered.wait(5)
    assert not group._initialization.done()
    group.start(session)

    def close():
        group.close()
        closed.set()

    closer = Thread(target=close)
    closer.start()
    try:
        assert not closed.wait(0.05)
    finally:
        release.set()
        closer.join(5)
    assert closed.is_set()
    group.close()
    assert native.owner_close.call_count == (0 if fails else 1)
    assert group.executor is group._initializer is group._initialization is None


def test_summary_sums_uneven_rank_bytes_and_uses_slowest_transfer():
    from b12x.loader._shared_checkpoint import SharedReadGroup

    members = [dict(destination_bytes=payload, local_io=dict(
        payload_bytes=payload, physical_bytes=before,
    )) for payload, before in zip((100, 120, 80, 90), (7, 11, 19, 23), strict=True)]
    reports = [dict(destination_bytes=member["destination_bytes"], physical_bytes=physical,
                    execution_seconds=seconds, gds_version=1150)
               for member, physical, seconds in zip(members, (31, 37, 41, 43), (1.5, 2.2, 1.8, 1.6), strict=True)]
    group = object.__new__(SharedReadGroup)
    group.failed = group.unsafe = False
    group.executor = None
    group.totals, group.summary, group.epoch, group.world_size = {}, {}, 0, 4
    group._phase = lambda name, action: (action(), [])
    group._prepare = Mock(return_value=members[0])
    group._plan = Mock(return_value=None)
    group._execute = Mock(return_value=reports[0])
    group._gather = Mock(side_effect=[members, reports])
    group.finish(SimpleNamespace(records=[], destinations=[]))
    assert group.summary == dict(ranks=4, payload_bytes=390, physical_bytes=212,
                                 shared_physical_bytes=152,
                                 shared_transfer_seconds=2.2)


def _worker(rank, rendezvous, checkpoint, output, inject_failure, world_size):
    from b12x.loader._checkpoint import DirectWeightSession

    from b12x.loader._shared_checkpoint import SharedReadGroup

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=90))
    group = SharedReadGroup(dist.group.WORLD, rank)
    if inject_failure == "initialization" and rank == 2:
        from b12x.loader._gds_native import load

        def fail_initialization(*args):
            raise RuntimeError("injected shared reader initialization failure")

        load().owner_create = fail_initialization
    with DirectWeightSession(rank, shared_read_group=group) as session:
        sources = dict(session.weights([checkpoint], prefixes=("bytes", "bf16")))
        prefix = (1 << 32) + 65537
        backing = torch.empty(prefix + 8192 * 1026 + 1, dtype=torch.uint8, device=rank)
        rows = backing[prefix:-1].view(8192, 1026)
        target = rows[:, 1:-1]
        expanded = torch.full((256, 66), -17, dtype=torch.float32, device=rank)
        backing[prefix - 1] = 199
        rows.fill_(199)
        backing[-1] = 199
        expected = torch.arange(256, dtype=torch.uint8).repeat(4).expand(8192, -1)
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(256, 256)
        group.start(session)
        session(target, sources["bytes"][:, rank * 1024:(rank + 1) * 1024])
        session(expanded[:, 1:-1], sources["bf16"][:, rank * 64:(rank + 1) * 64])
        if inject_failure:
            if inject_failure == "read":
                execute = group._execute

                def truncated_read(plan):
                    if rank == 2:
                        os.truncate(checkpoint, 4096)
                    dist.barrier()
                    return execute(plan)

                group._execute = truncated_read
            if rank == 2:
                if inject_failure == "destination":
                    session.records[3] = 1
                elif inject_failure == "source":
                    session.records[1] = (1 << 40)
                elif inject_failure == "overlap":
                    session.records.extend(session.records[:8])
                elif inject_failure == "identity":
                    stat = os.stat(checkpoint)
                    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            with pytest.raises(RuntimeError, match="shared checkpoint completion failed"):
                session.finish()
            assert group.failed
            assert torch.all(target == 199)
            assert torch.all(expanded == -17)
            with pytest.raises(RuntimeError, match="group failed"):
                session.flush()
        else:
            # A rank-local numerical consumer must not introduce a collective.
            if rank == 0:
                session.flush()
            if rank == 3:
                time.sleep(0.2)
            session.finish()
            assert torch.equal(target.cpu(), expected)
            assert torch.equal(expanded[:, 1:-1].cpu().view(torch.int32),
                               bits[:, rank * 64:(rank + 1) * 64].to(torch.int32) << 16)
            assert torch.all(rows[:, 0] == 199) and torch.all(rows[:, -1] == 199)
            assert backing[prefix - 1].item() == backing[-1].item() == 199
            assert torch.all(expanded[:, 0] == -17) and torch.all(expanded[:, -1] == -17)
            target.bitwise_not_()
            session(target, sources["bytes"][:, rank * 1024:(rank + 1) * 1024])
            session.finish()
            assert torch.equal(target.cpu(), expected)
            totals = dict(group.totals)
    if not inject_failure:
        pointer = target.data_ptr()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            captured = target[:16].clone()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured.copy_(target[:16])
        for _ in range(3):
            captured.zero_()
            graph.replay()
            assert torch.equal(captured.cpu(), expected[:16])
        assert target.data_ptr() == pointer
        Path(output, f"rank-{rank}.json").write_text(json.dumps(totals))
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size,inject_failure", [
    (4, failure) for failure in (None, "initialization", "destination", "source", "overlap", "identity", "read")
] + [(1, None)])
def test_shared_gds_completion_and_group_error(tmp_path, world_size, inject_failure):
    if torch.cuda.device_count() != 4:
        pytest.skip("select exactly four assigned CUDA devices for shared GDS qualification")
    offset = (1 << 32) + 4104
    payload = torch.arange(256, dtype=torch.uint8).repeat(8192 * 16).numpy().tobytes()
    bf16 = torch.arange(65536, dtype=torch.int32).to(torch.int16).numpy().tobytes()
    header = json.dumps({
        "padding": {"dtype": "U8", "shape": [offset - 4096], "data_offsets": [0, offset - 4096]},
        "bytes": {"dtype": "U8", "shape": [8192, 4096], "data_offsets": [offset - 4096, offset - 4096 + len(payload)]},
        "bf16": {"dtype": "BF16", "shape": [256, 256], "data_offsets": [offset - 4096 + len(payload), offset - 4096 + len(payload) + len(bf16)]},
    }).encode().ljust(4088, b" ")
    checkpoint = tmp_path / "weights.safetensors"
    with checkpoint.open("wb") as file:
        file.write(len(header).to_bytes(8, "little") + header)
        file.seek(offset)
        file.write(payload)
        file.write(bf16)
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(checkpoint), str(tmp_path), inject_failure, world_size), nprocs=world_size, join=True)
    if not inject_failure:
        results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(world_size)]
        assert all(result["epochs"] == 2 for result in results)
        assert all(result["physical_bytes"] > 0 for result in results)
        assert sum(result["physical_bytes"] for result in results) < 2 * (len(payload) + len(bf16) + 8192)
