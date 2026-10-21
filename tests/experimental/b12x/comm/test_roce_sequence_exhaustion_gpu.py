"""Check the native proxy's generation limit with real GPU graph replay.

Run with one torchrun process per RoCE host. A test-only C wrapper initializes
the stopped proxy near its generation limit; production transport interfaces
and layouts remain unchanged.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from tests.experimental.b12x.comm.test_roce_varlen_gpu import _prepare

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires torchrun with WORLD_SIZE >= 2",
)


def _seed_proxy_library(tmp_path):
    from b12x.comm.roce import _proxy

    source = Path(_proxy.__file__).with_name("_roce_proxy.c")
    wrapper = tmp_path / "seed_proxy.c"
    wrapper.write_text(
        f'#include "{source}"\n'
        "int test_seed_stopped_proxy(roce_ctx_t *c, uint32_t sequence) {\n"
        "  if (atomic_load(&c->running)) return -1;\n"
        "  c->last_seq = sequence;\n"
        "  c->started = 1;\n"
        "  return 0;\n"
        "}\n"
    )
    library = tmp_path / "seed_proxy.so"
    subprocess.run(
        [
            _proxy._compiler(),
            "-O2",
            "-std=gnu11",
            "-shared",
            "-fPIC",
            str(wrapper),
            "-o",
            str(library),
            "-libverbs",
            "-lpthread",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = ctypes.CDLL(str(library))
    result.test_seed_stopped_proxy.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    result.test_seed_stopped_proxy.restype = ctypes.c_int
    return result


@pytest.mark.parametrize("ring", [False, True], ids=["mesh", "ring"])
def test_proxy_rejects_generation_after_last_valid_graph_replay(tmp_path, ring):
    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    seed_library = _seed_proxy_library(tmp_path)
    prepared = _prepare(spin_limit=2_000_000, ring=ring)
    runtime = prepared.runtime
    rank, world = dist.get_rank(), dist.get_world_size()
    last_valid = (1 << 31) - 1
    source = torch.full(
        (1 << 20,), rank + 1, dtype=torch.bfloat16, device=runtime.device
    )
    output = torch.empty_like(source)
    stream = torch.cuda.Stream(device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            prepared.all_reduce(source, out=output)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), runtime.capture(stream=stream):
            prepared.all_reduce(source, out=output)
        torch.cuda.synchronize()
        dist.barrier()
        runtime._proxy.stop()
        runtime._region.zero_()
        runtime._counters.zero_()
        runtime._counters[0] = last_valid - 1
        runtime._ctrl_words[0] = last_valid - 1
        torch.cuda.synchronize()
        assert (
            seed_library.test_seed_stopped_proxy(runtime._proxy._ctx, last_valid - 1)
            == 0
        )
        dist.barrier()
        runtime._proxy.start()
        posted = runtime._proxy.stats()["ops_posted"]
        graph.replay()
        torch.cuda.synchronize()
        runtime.check_health()
        assert int(runtime._counters[0].item()) == last_valid
        assert torch.equal(output, torch.full_like(output, world * (world + 1) // 2))
        assert runtime._proxy.stats()["ops_posted"] == posted + 1
        source.fill_(rank + 2)
        output.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        with pytest.raises(RuntimeError, match="sequence space exhausted.*recreate"):
            runtime.check_health()
        assert int(runtime._counters[0].item()) == last_valid
        assert runtime._proxy.stats()["ops_posted"] == posted + 1
        expected = torch.full_like(output, world * (world + 1) // 2 + world)
        assert not torch.equal(output, expected)
    finally:
        runtime.close()
        dist.barrier()
