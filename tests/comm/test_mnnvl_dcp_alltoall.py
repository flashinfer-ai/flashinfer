# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Multi-GPU tests for DCP LL128 FIFO All-to-All with MNNVL workspace.

Complements test_dcp_alltoall.py (single-GPU simulation) by running
real multi-process, multi-GPU alltoall via MPI. This catches bugs
that single-GPU tests cannot:
  - MNNVL workspace allocation and communicator grouping
  - Cross-GPU memory visibility (LL128 FIFO writes to peer memory)
  - Workspace shape [cp_size, ws_elems] with real multi-rank segments

Run:
  mpirun -np 2 pytest tests/comm/test_mnnvl_dcp_alltoall.py -v -s
  mpirun -np 4 pytest tests/comm/test_mnnvl_dcp_alltoall.py -v -s
"""

import socket

import pynvml
import pytest
import torch

from flashinfer.comm import (
    decode_cp_a2a_alltoall,
    decode_cp_a2a_allocate_workspace,
    decode_cp_a2a_init_workspace,
    decode_cp_a2a_workspace_size,
)
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlMemory, MpiComm

from .conftest import mnnvl_available

pynvml.nvmlInit()


# ─── SM90+ gate ──────────────────────────────────────────────────────────


def _dcp_alltoall_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    from flashinfer.utils import get_compute_capability

    major, _ = get_compute_capability(torch.device("cuda"))
    return major in (9, 10, 11, 12)


def _mpi4py_available() -> bool:
    try:
        import mpi4py  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _dcp_alltoall_supported(),
        reason="Requires SM90+ GPU (Hopper or Blackwell family)",
    ),
    pytest.mark.skipif(
        not mnnvl_available(),
        reason="MNNVL not supported on this platform or container lacks SYS_PTRACE",
    ),
    pytest.mark.skipif(
        not _mpi4py_available(),
        reason="mpi4py not installed (run with: mpirun -np N pytest ...)",
    ),
]


# ─── Helper ──────────────────────────────────────────────────────────────


def _to_torch(t):
    """Convert a tvm_ffi.core.Tensor (or any DLPack object) to torch.Tensor."""
    if isinstance(t, torch.Tensor):
        return t
    return torch.from_dlpack(t)


def _setup_rank():
    """Initialize MPI rank and CUDA device. Returns (rank, world_size, comm)."""
    comm = MpiComm()
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Get local rank from hostname
    hostname = socket.gethostname()
    all_hostnames = comm.allgather(hostname)
    local_ranks_before_me = sum(1 for i in range(rank) if all_hostnames[i] == hostname)
    local_rank = local_ranks_before_me
    torch.cuda.set_device(local_rank)

    return rank, world_size, comm


# ─── Module-level MNNVL workspace (allocated once, reused across tests) ──

# Guard module-level allocation: pytestmark skipif conditions are not
# enforced during module import (collection phase). If we allocate
# unconditionally, CI environments without SYS_PTRACE will fail at
# collection time instead of gracefully skipping.
if _dcp_alltoall_supported() and mnnvl_available() and _mpi4py_available():
    _rank, _cp_size, _comm = _setup_rank()

    def _allocate_mnnvl_workspace_once():
        """Allocate MNNVL workspace once at module level.

        MnnvlMemory uses a global bump allocator that doesn't support
        individual frees. Allocating per-test causes segfaults when
        workspace tensors from previous tests get GC'd. So we allocate
        once and reuse.
        """
        MnnvlMemory.initialize()
        MnnvlMemory.comm = _comm

        mapping = Mapping(
            world_size=_cp_size,
            rank=_rank,
            cp_size=_cp_size,
            tp_size=1,
            pp_size=1,
        )

        ws_bytes = decode_cp_a2a_workspace_size(_cp_size)
        mnnvl_mem = MnnvlMemory(mapping, ws_bytes)
        workspace = mnnvl_mem.as_torch_strided_tensor(torch.int64)
        workspace._mnnvl_mem = mnnvl_mem  # prevent GC
        return workspace

    _mnnvl_workspace = _allocate_mnnvl_workspace_once()
else:
    _rank, _cp_size, _comm = 0, 1, None
    _mnnvl_workspace = None


# ─── Tests ───────────────────────────────────────────────────────────────


class TestMnnvlDcpWorkspace:
    """Test MNNVL workspace allocation for DCP A2A."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0xA2A + _rank)
        yield

    def test_workspace_shape(self):
        """MNNVL workspace must have shape [cp_size, ws_elems_per_rank]."""
        try:
            assert _mnnvl_workspace.shape[0] == _cp_size, (
                f"Expected workspace.shape[0] == {_cp_size}, got {_mnnvl_workspace.shape[0]}"
            )

            ws_bytes = decode_cp_a2a_workspace_size(_cp_size)
            expected_elems = (ws_bytes + 7) // 8  # int64 elements
            assert _mnnvl_workspace.shape[1] == expected_elems
            assert _mnnvl_workspace.dtype == torch.int64
        finally:
            _comm.Barrier()

    def test_workspace_cross_rank_visible(self):
        """Each rank can write to its own segment and peers can read it."""
        try:
            # Each rank writes a unique pattern to its own workspace segment
            pattern = torch.full_like(_mnnvl_workspace[_rank], fill_value=_rank + 1)
            _mnnvl_workspace[_rank].copy_(pattern)
            torch.cuda.synchronize()
            _comm.Barrier()

            # Each rank reads all segments and verifies the pattern
            for peer in range(_cp_size):
                expected = peer + 1
                actual = _mnnvl_workspace[peer][0].item()
                assert actual == expected, (
                    f"Rank {_rank}: workspace[{peer}][0] = {actual}, expected {expected}"
                )
        finally:
            _comm.Barrier()


class TestMnnvlDcpAlltoall:
    """Multi-GPU correctness tests for DCP A2A alltoall."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0xA2A)
        yield

    def _run_alltoall(self, batch_size, head_dim, stats_dim, dtype):
        """Run DCP A2A alltoall on real multi-GPU and verify correctness.

        The transpose property must hold:
          recv_o[rank][.., peer, :] == send_o[peer][.., rank, :]
          recv_s[rank][.., peer, :] == send_s[peer][.., rank, :]
        """
        try:
            workspace = _mnnvl_workspace

            decode_cp_a2a_init_workspace(workspace, _rank, _cp_size)
            torch.cuda.synchronize()
            _comm.Barrier()

            # Generate input with deterministic seed per rank
            torch.manual_seed(0xA2A + _rank)
            partial_o = torch.randn(
                batch_size, _cp_size, head_dim, dtype=dtype, device="cuda"
            )
            softmax_stats = torch.randn(
                batch_size, _cp_size, stats_dim, dtype=torch.float32, device="cuda"
            )

            # Run alltoall
            recv_o, recv_s = decode_cp_a2a_alltoall(
                partial_o, softmax_stats, workspace, _rank, _cp_size
            )
            recv_o = _to_torch(recv_o)
            recv_s = _to_torch(recv_s)
            torch.cuda.synchronize()
            _comm.Barrier()

            # Gather all inputs to all ranks for verification
            all_partial_o = _comm.allgather(partial_o.cpu())
            all_softmax_stats = _comm.allgather(softmax_stats.cpu())

            # Verify transpose property
            for peer in range(_cp_size):
                expected_o = all_partial_o[peer][..., _rank, :].cuda()
                expected_s = all_softmax_stats[peer][..., _rank, :].cuda()

                torch.testing.assert_close(
                    recv_o[..., peer, :],
                    expected_o,
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    recv_s[..., peer, :],
                    expected_s,
                    atol=0,
                    rtol=0,
                )
        finally:
            _comm.Barrier()

    @pytest.mark.parametrize(
        "batch_size,head_dim,stats_dim,dtype",
        [
            pytest.param(1, 128, 2, torch.bfloat16, id="B1-D128-S2-bf16"),
            pytest.param(16, 128, 2, torch.bfloat16, id="B16-D128-S2-bf16"),
            pytest.param(128, 128, 2, torch.bfloat16, id="B128-D128-S2-bf16"),
            pytest.param(16, 256, 4, torch.bfloat16, id="B16-D256-S4-bf16"),
            pytest.param(16, 128, 2, torch.float16, id="B16-D128-S2-fp16"),
        ],
    )
    def test_alltoall_correctness(self, batch_size, head_dim, stats_dim, dtype):
        """Verify transpose property across real GPUs."""
        self._run_alltoall(batch_size, head_dim, stats_dim, dtype)

    def test_repeated_alltoall(self):
        """Multiple alltoall calls on the same workspace (FIFO reuse)."""
        try:
            workspace = _mnnvl_workspace

            decode_cp_a2a_init_workspace(workspace, _rank, _cp_size)
            torch.cuda.synchronize()
            _comm.Barrier()

            for round_idx in range(3):
                torch.manual_seed(0xA2A + _rank * 100 + round_idx)
                partial_o = torch.randn(
                    16, _cp_size, 128, dtype=torch.bfloat16, device="cuda"
                )
                softmax_stats = torch.randn(
                    16, _cp_size, 2, dtype=torch.float32, device="cuda"
                )

                recv_o, recv_s = decode_cp_a2a_alltoall(
                    partial_o, softmax_stats, workspace, _rank, _cp_size
                )
                recv_o = _to_torch(recv_o)
                recv_s = _to_torch(recv_s)
                torch.cuda.synchronize()
                _comm.Barrier()

                all_partial_o = _comm.allgather(partial_o.cpu())
                all_softmax_stats = _comm.allgather(softmax_stats.cpu())

                for peer in range(_cp_size):
                    torch.testing.assert_close(
                        recv_o[..., peer, :],
                        all_partial_o[peer][..., _rank, :].cuda(),
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        recv_s[..., peer, :],
                        all_softmax_stats[peer][..., _rank, :].cuda(),
                        atol=0,
                        rtol=0,
                    )
        finally:
            _comm.Barrier()


class TestMnnvlDcpDeviceMemoryFallback:
    """Test that non-MNNVL (device memory) path also works multi-GPU.

    Uses decode_cp_a2a_allocate_workspace without MNNVL mapping. This only
    works when all ranks are on the same GPU (single-GPU simulation)
    or with IPC. Included here to verify the workspace API contract.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0xA2A)
        yield

    def test_device_workspace_shape(self):
        """Device workspace has correct shape [cp_size, ws_elems]."""
        try:
            workspace = decode_cp_a2a_allocate_workspace(_cp_size, cp_rank=_rank)
            assert workspace.shape[0] == _cp_size

            ws_bytes = decode_cp_a2a_workspace_size(_cp_size)
            expected_elems = (ws_bytes + 7) // 8
            assert workspace.shape[1] == expected_elems
            assert workspace.dtype == torch.int64
        finally:
            _comm.Barrier()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
