"""GPU graph replay on a native neighbor ring (torchrun, 2..16 ranks).

Unlike a physical cabling test, this can run on a switched Spark cluster:
non-neighbor queue pairs are left unconnected by construction. The CPU control
group is Gloo, so this test does not allocate an NCCL communicator.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from tests.experimental.b12x.comm.test_roce_twoshot_gpu import _prepare

pytestmark = pytest.mark.skipif(
    not 2 <= int(os.environ.get("WORLD_SIZE", "1")) <= 16,
    reason="requires torchrun with 2..16 ranks",
)


def test_ring_replays_mixed_collectives_with_changed_inputs():
    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    prepared = _prepare(ring=True)
    runtime = prepared.runtime
    rank, world = dist.get_rank(), dist.get_world_size()
    neighbors = {(rank - 1) % world, (rank + 1) % world}
    assert {p for p, links in enumerate(runtime.routes) if links} == neighbors
    assert runtime.native_description()["ring"] is True
    small = torch.full((16384,), rank + 1, dtype=torch.bfloat16, device=runtime.device)
    large = torch.full(
        (3 << 19,), rank + 1, dtype=torch.bfloat16, device=runtime.device
    )
    small_out, large_out = torch.empty_like(small), torch.empty_like(large)
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            prepared.all_reduce(small, out=small_out)
            prepared.all_reduce(large, out=large_out)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
            prepared.all_reduce(large, out=large_out)
            prepared.all_reduce(small, out=small_out)
            prepared.all_reduce(large, out=large_out)
        for iteration in range(20):
            small.fill_(rank + 1 + iteration)
            large.fill_(rank + 1 + iteration)
            small_out.zero_()
            large_out.zero_()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            runtime.check_health()
            value = world * (world + 1) // 2 + world * iteration
            assert torch.equal(small_out, torch.full_like(small_out, value))
            assert torch.equal(large_out, torch.full_like(large_out, value))
        assert runtime._proxy.stats()["connected_peers"] == min(2, world - 1)
        dist.barrier()
    finally:
        runtime.close()
