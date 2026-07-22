# Copyright (c) 2024 by FlashInfer team.
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
"""CPU-only coverage for PosixFDHandleExchanger (the SCM_RIGHTS fd exchange that
MnnvlMemory's POSIX-FD path depends on). No GPU or NVLink required: each rank
shares a memfd tagged with its rank and we assert the gathered fds map to the
correct source ranks -- catching any ring-ordering or indexing regression.
"""
import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from flashinfer.comm.mnnvl import PosixFDHandleExchanger, TorchDistBackend


def _worker(rank: int, world: int, init_method: str, q):
    tag = f"rank-{rank}".encode()
    try:
        dist.init_process_group(
            "gloo", rank=rank, world_size=world, init_method=init_method
        )
        # A memfd whose contents identify this rank; shareable via SCM_RIGHTS.
        local_fd = os.memfd_create(f"posixfd-test-{rank}", 0)
        os.write(local_fd, tag)

        exchanger = PosixFDHandleExchanger(TorchDistBackend(), rank, world)
        try:
            gathered = exchanger.allgather(local_fd)
        finally:
            exchanger.close()

        # gathered[src] must be a local fd referring to rank `src`'s memfd.
        ok = True
        for src, rfd in enumerate(gathered):
            if os.pread(rfd, 32, 0) != f"rank-{src}".encode():
                ok = False
            os.close(rfd)
        os.close(local_fd)
        q.put((rank, ok, None))
    except Exception as e:  # noqa: BLE001 - report to parent for a clean assert
        q.put((rank, False, repr(e)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2, 4])
def test_posix_fd_exchanger_allgather(world: int):
    if not hasattr(os, "memfd_create"):
        pytest.skip("memfd_create requires Linux + Python >= 3.8")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init_method = f"tcp://127.0.0.1:{29600 + world}"
    procs = [
        ctx.Process(target=_worker, args=(r, world, init_method, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    try:
        outs = [q.get(timeout=120) for _ in range(world)]
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
    for rank, ok, err in sorted(outs):
        assert ok, f"rank {rank} fd-exchange failed: {err}"
