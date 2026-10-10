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

# Helpers shared by the Ulysses test modules and tests/comm/conftest.py:
# rendezvous paths, the fake full-mesh topology and the IPC/JIT trap.

import contextlib
import importlib
import os
import tempfile


def fresh_rendezvous_path():
    """A unique, unused path for a FileStore rendezvous.

    Not a TCP port: binding and closing a socket to pick one leaves a race,
    and each test starts eight workers.
    """
    handle, path = tempfile.mkstemp(prefix="flashinfer_ulysses_pg_")
    os.close(handle)
    # torch's FileStore wants to create the file itself.
    os.unlink(path)
    return path


@contextlib.contextmanager
def rendezvous_path():
    path = fresh_rendezvous_path()
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def full_mesh(world_size, hostname="hostA"):
    """A fake single-host topology where every pair is P2P- and NVLink-connected."""
    from flashinfer.comm.ulysses_topology import UlyssesRankTopology

    uuids = [f"GPU-fake-{i}" for i in range(world_size)]
    return [
        UlyssesRankTopology(
            rank=r,
            hostname=hostname,
            device_index=r,
            device_uuid=uuids[r],
            pci_bus_id=f"0000:{r:02x}:00.0",
            peer_p2p={uuids[p]: True for p in range(world_size) if p != r},
            peer_nvlink={uuids[p]: True for p in range(world_size) if p != r},
        )
        for r in range(world_size)
    ]


def forbid_ipc_and_jit(monkeypatch):
    """Trap every IPC allocation and JIT entry point a host-only test must not reach."""
    cuda_ipc_mod = importlib.import_module("flashinfer.comm.cuda_ipc")
    ulysses_mod = importlib.import_module("flashinfer.comm.ulysses")
    vllm_ar_mod = importlib.import_module("flashinfer.comm.vllm_ar")

    def _boom(*args, **kwargs):
        raise AssertionError("IPC/JIT entry point must not be touched")

    monkeypatch.setattr(cuda_ipc_mod, "create_shared_buffer", _boom)
    monkeypatch.setattr(cuda_ipc_mod.cudart, "cudaMalloc", _boom, raising=False)
    monkeypatch.setattr(ulysses_mod, "get_ulysses_a2a_module", _boom)
    monkeypatch.setattr(ulysses_mod, "init_ulysses_a2a", _boom)
    monkeypatch.setattr(vllm_ar_mod, "meta_size", _boom)
    # the merged module binds gen_ulysses_a2a_module at import: patch the
    # local binding, not flashinfer.jit.comm (which would not intercept)
    monkeypatch.setattr(ulysses_mod, "gen_ulysses_a2a_module", _boom)
