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

"""Tests for flashinfer.comm.dcp_alltoall — DCP LL128 FIFO All-to-All.

Single-GPU multi-rank pattern: simulates cp_size ranks on one GPU using
separate CUDA streams for the alltoall phase. All ranks share a single
workspace tensor of shape [cp_size, ws_elems_per_rank].

Run: python -m pytest tests/comm/test_dcp_alltoall.py -v -s
"""

import pytest
import torch

from flashinfer.comm import (
    decode_cp_a2a_alltoall,
    decode_cp_a2a_allocate_workspace,
    decode_cp_a2a_init_workspace,
    decode_cp_a2a_workspace_size,
)


# ─── SM90/SM100 gate ─────────────────────────────────────────────────────


def _dcp_alltoall_supported() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        # DCP A2A requires TMA + mbarrier + PDL (SM90 baseline); the kernel
        # builds for SM90 / SM10x / SM11x / SM12x.
        return major in (9, 10, 11, 12)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _dcp_alltoall_supported(),
    reason="Requires SM90+ GPU (Hopper or Blackwell family)",
)


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Set torch seed for deterministic tests."""
    torch.manual_seed(0xA2A)
    yield


# ─── Helper ──────────────────────────────────────────────────────────────


def _to_torch(t):
    """Convert a tvm_ffi.core.Tensor (or any DLPack object) to torch.Tensor."""
    if isinstance(t, torch.Tensor):
        return t
    return torch.from_dlpack(t)


def _run_single_gpu_alltoall(cp_size, batch_size, head_dim, stats_dim, dtype):
    """Simulate cp_size ranks on one GPU and return (inputs, outputs, workspace).

    1. Allocate shared workspace (plain device memory).
    2. Generate random partial_o [B, cp_size, D] and softmax_stats [B, cp_size, S]
       per rank.
    3. Init workspace for all ranks (default stream, sequential).
    4. torch.cuda.synchronize() — cross-rank barrier.
    5. Alltoall for each rank on separate CUDA streams.
    6. Sync all streams and return results.
    """
    torch.cuda.set_device(0)

    workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)

    all_partial_o = []
    all_softmax_stats = []
    for _ in range(cp_size):
        po = torch.randn(batch_size, cp_size, head_dim, dtype=dtype, device="cuda")
        ss = torch.randn(
            batch_size, cp_size, stats_dim, dtype=torch.float32, device="cuda"
        )
        all_partial_o.append(po)
        all_softmax_stats.append(ss)

    for r in range(cp_size):
        decode_cp_a2a_init_workspace(workspace, r, cp_size)

    torch.cuda.synchronize()

    streams = [torch.cuda.Stream() for _ in range(cp_size)]
    recv_o = [None] * cp_size
    recv_s = [None] * cp_size

    for r in range(cp_size):
        with torch.cuda.stream(streams[r]):
            o, s = decode_cp_a2a_alltoall(
                all_partial_o[r],
                all_softmax_stats[r],
                workspace,
                r,
                cp_size,
            )
            recv_o[r] = _to_torch(o)
            recv_s[r] = _to_torch(s)

    for stream in streams:
        stream.synchronize()
    torch.cuda.synchronize()

    return all_partial_o, all_softmax_stats, recv_o, recv_s, workspace


def _verify_transpose(cp_size, all_partial_o, all_softmax_stats, recv_o, recv_s):
    """Assert the transpose property for all (rank, peer) pairs.

    recv_o[r][..., peer, :] == all_partial_o[peer][..., r, :]
    recv_s[r][..., peer, :] == all_softmax_stats[peer][..., r, :]
    """
    for r in range(cp_size):
        for peer in range(cp_size):
            torch.testing.assert_close(
                recv_o[r][..., peer, :],
                all_partial_o[peer][..., r, :],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                recv_s[r][..., peer, :],
                all_softmax_stats[peer][..., r, :],
                atol=0,
                rtol=0,
            )


# ─── Workspace Lifecycle Tests ───────────────────────────────────────────


class TestWorkspaceLifecycle:
    """Verify workspace sizing, allocation, and initialization."""

    def test_workspace_size_positive(self):
        for cp_size in [2, 4, 8]:
            ws = decode_cp_a2a_workspace_size(cp_size)
            assert isinstance(ws, int)
            assert ws > 0, f"cp_size={cp_size}: workspace_size should be positive"

    def test_workspace_size_monotonic(self):
        ws2 = decode_cp_a2a_workspace_size(2)
        ws4 = decode_cp_a2a_workspace_size(4)
        ws8 = decode_cp_a2a_workspace_size(8)
        assert ws4 > ws2, "ws(4) should be > ws(2)"
        assert ws8 > ws4, "ws(8) should be > ws(4)"

    def test_allocate_returns_correct_shape_and_dtype(self):
        for cp_size in [2, 4]:
            ws_bytes = decode_cp_a2a_workspace_size(cp_size)
            workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)
            assert workspace.dtype == torch.int64
            assert workspace.shape[0] == cp_size
            assert workspace.shape[1] == (ws_bytes + 7) // 8

    def test_init_workspace_does_not_hang(self):
        for cp_size in [2, 4]:
            workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)
            for r in range(cp_size):
                decode_cp_a2a_init_workspace(workspace, r, cp_size)
            torch.cuda.synchronize()


# ─── Correctness Tests (Single-GPU Multi-Rank) ──────────────────────────

CORRECTNESS_PARAMS = [
    # cp_size=2 (minimum)
    pytest.param(2, 1, 128, 2, torch.bfloat16, id="cp2-B1-D128-S2-bf16"),
    pytest.param(2, 16, 128, 2, torch.bfloat16, id="cp2-B16-D128-S2-bf16"),
    pytest.param(2, 128, 128, 2, torch.bfloat16, id="cp2-B128-D128-S2-bf16"),
    pytest.param(2, 16, 256, 2, torch.bfloat16, id="cp2-B16-D256-S2-bf16"),
    pytest.param(2, 16, 128, 4, torch.bfloat16, id="cp2-B16-D128-S4-bf16"),
    pytest.param(2, 16, 128, 2, torch.float16, id="cp2-B16-D128-S2-fp16"),
    # cp_size=4 (common single-node)
    pytest.param(4, 1, 128, 2, torch.bfloat16, id="cp4-B1-D128-S2-bf16"),
    pytest.param(4, 16, 128, 2, torch.bfloat16, id="cp4-B16-D128-S2-bf16"),
    pytest.param(4, 128, 128, 2, torch.bfloat16, id="cp4-B128-D128-S2-bf16"),
    pytest.param(4, 16, 256, 4, torch.bfloat16, id="cp4-B16-D256-S4-bf16"),
    pytest.param(4, 16, 128, 2, torch.float16, id="cp4-B16-D128-S2-fp16"),
    pytest.param(4, 16, 256, 2, torch.float16, id="cp4-B16-D256-S2-fp16"),
    # cp_size=8 (full single-node, e.g. 8xH200)
    pytest.param(8, 1, 128, 2, torch.bfloat16, id="cp8-B1-D128-S2-bf16"),
    pytest.param(8, 16, 128, 2, torch.bfloat16, id="cp8-B16-D128-S2-bf16"),
    pytest.param(8, 64, 128, 2, torch.bfloat16, id="cp8-B64-D128-S2-bf16"),
    pytest.param(8, 16, 256, 4, torch.bfloat16, id="cp8-B16-D256-S4-bf16"),
    pytest.param(8, 16, 128, 2, torch.float16, id="cp8-B16-D128-S2-fp16"),
]


@pytest.mark.parametrize(
    "cp_size,batch_size,head_dim,stats_dim,dtype",
    CORRECTNESS_PARAMS,
)
def test_alltoall_correctness(cp_size, batch_size, head_dim, stats_dim, dtype):
    """Verify the transpose property: recv[r][.., peer, :] == input[peer][.., r, :]."""
    all_po, all_ss, recv_o, recv_s, _ = _run_single_gpu_alltoall(
        cp_size, batch_size, head_dim, stats_dim, dtype
    )
    _verify_transpose(cp_size, all_po, all_ss, recv_o, recv_s)


# ─── FIFO Reuse (Repeated Alltoall without re-init) ─────────────────────

FIFO_REUSE_PARAMS = [
    pytest.param(2, 16, 128, 2, torch.bfloat16, 3, id="cp2-3rounds"),
    pytest.param(4, 16, 128, 2, torch.bfloat16, 3, id="cp4-3rounds"),
]


@pytest.mark.parametrize(
    "cp_size,batch_size,head_dim,stats_dim,dtype,num_rounds",
    FIFO_REUSE_PARAMS,
)
def test_repeated_alltoall(cp_size, batch_size, head_dim, stats_dim, dtype, num_rounds):
    """Multiple alltoall calls on the same workspace without re-init (FIFO reuse)."""
    torch.cuda.set_device(0)

    workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)

    for r in range(cp_size):
        decode_cp_a2a_init_workspace(workspace, r, cp_size)
    torch.cuda.synchronize()

    for _round_idx in range(num_rounds):
        all_po = [
            torch.randn(batch_size, cp_size, head_dim, dtype=dtype, device="cuda")
            for _ in range(cp_size)
        ]
        all_ss = [
            torch.randn(
                batch_size, cp_size, stats_dim, dtype=torch.float32, device="cuda"
            )
            for _ in range(cp_size)
        ]

        streams = [torch.cuda.Stream() for _ in range(cp_size)]
        recv_o = [None] * cp_size
        recv_s = [None] * cp_size

        for r in range(cp_size):
            with torch.cuda.stream(streams[r]):
                o, s = decode_cp_a2a_alltoall(
                    all_po[r], all_ss[r], workspace, r, cp_size
                )
                recv_o[r] = _to_torch(o)
                recv_s[r] = _to_torch(s)

        for stream in streams:
            stream.synchronize()
        torch.cuda.synchronize()

        _verify_transpose(cp_size, all_po, all_ss, recv_o, recv_s)


# ─── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: cp_size=1, batch_size=0."""

    def test_cp_size_1_is_identity(self):
        """cp_size=1: output should equal input (no peers to exchange with)."""
        cp_size, batch_size, head_dim, stats_dim = 1, 16, 128, 2
        dtype = torch.bfloat16
        torch.cuda.set_device(0)

        workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)
        po = torch.randn(batch_size, cp_size, head_dim, dtype=dtype, device="cuda")
        ss = torch.randn(
            batch_size, cp_size, stats_dim, dtype=torch.float32, device="cuda"
        )

        decode_cp_a2a_init_workspace(workspace, 0, cp_size)
        torch.cuda.synchronize()

        o, s = decode_cp_a2a_alltoall(po, ss, workspace, 0, cp_size)
        o = _to_torch(o)
        s = _to_torch(s)

        torch.testing.assert_close(o, po, atol=0, rtol=0)
        torch.testing.assert_close(s, ss, atol=0, rtol=0)

    def test_batch_size_0(self):
        """batch_size=0: should not crash (zero entries → no work)."""
        cp_size, batch_size, head_dim, stats_dim = 2, 0, 128, 2
        dtype = torch.bfloat16
        torch.cuda.set_device(0)

        workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)
        po = torch.randn(batch_size, cp_size, head_dim, dtype=dtype, device="cuda")
        ss = torch.randn(
            batch_size, cp_size, stats_dim, dtype=torch.float32, device="cuda"
        )

        for r in range(cp_size):
            decode_cp_a2a_init_workspace(workspace, r, cp_size)
        torch.cuda.synchronize()

        # Should not crash; output shape should match input shape
        o, s = decode_cp_a2a_alltoall(po, ss, workspace, 0, cp_size)
        o = _to_torch(o)
        s = _to_torch(s)
        assert o.shape == po.shape
        assert s.shape == ss.shape


# ─── Input Validation (Error Handling) ───────────────────────────────────


class TestInputValidation:
    """Verify that invalid inputs are rejected with errors, not silent corruption."""

    def test_wrong_dtype_float64(self):
        """partial_o with float64 should be rejected."""
        cp_size = 2
        workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)
        for r in range(cp_size):
            decode_cp_a2a_init_workspace(workspace, r, cp_size)
        torch.cuda.synchronize()

        po = torch.randn(16, cp_size, 128, dtype=torch.float64, device="cuda")
        ss = torch.randn(16, cp_size, 2, dtype=torch.float32, device="cuda")

        with pytest.raises(RuntimeError):
            decode_cp_a2a_alltoall(po, ss, workspace, 0, cp_size)

    def test_wrong_dtype_float32(self):
        """partial_o with float32 should be rejected (must be half/bfloat16)."""
        cp_size = 2
        workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)
        for r in range(cp_size):
            decode_cp_a2a_init_workspace(workspace, r, cp_size)
        torch.cuda.synchronize()

        po = torch.randn(16, cp_size, 128, dtype=torch.float32, device="cuda")
        ss = torch.randn(16, cp_size, 2, dtype=torch.float32, device="cuda")

        with pytest.raises(RuntimeError):
            decode_cp_a2a_alltoall(po, ss, workspace, 0, cp_size)

    def test_stats_dim_1_odd_alignment(self):
        """stats_dim=1 violates 'even and >= 2' constraint — should error."""
        cp_size = 2
        workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)
        for r in range(cp_size):
            decode_cp_a2a_init_workspace(workspace, r, cp_size)
        torch.cuda.synchronize()

        po = torch.randn(16, cp_size, 128, dtype=torch.bfloat16, device="cuda")
        ss = torch.randn(16, cp_size, 1, dtype=torch.float32, device="cuda")

        with pytest.raises(RuntimeError):
            decode_cp_a2a_alltoall(po, ss, workspace, 0, cp_size)

    def test_mismatched_batch_dims(self):
        """partial_o and softmax_stats with different batch sizes should error."""
        cp_size = 2
        workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)
        for r in range(cp_size):
            decode_cp_a2a_init_workspace(workspace, r, cp_size)
        torch.cuda.synchronize()

        po = torch.randn(16, cp_size, 128, dtype=torch.bfloat16, device="cuda")
        ss = torch.randn(32, cp_size, 2, dtype=torch.float32, device="cuda")

        with pytest.raises(RuntimeError):
            decode_cp_a2a_alltoall(po, ss, workspace, 0, cp_size)

    def test_wrong_stats_dtype(self):
        """softmax_stats with half instead of float32 should error."""
        cp_size = 2
        workspace = decode_cp_a2a_allocate_workspace(cp_size, cp_rank=0)
        for r in range(cp_size):
            decode_cp_a2a_init_workspace(workspace, r, cp_size)
        torch.cuda.synchronize()

        po = torch.randn(16, cp_size, 128, dtype=torch.bfloat16, device="cuda")
        ss = torch.randn(16, cp_size, 2, dtype=torch.float16, device="cuda")

        with pytest.raises(RuntimeError):
            decode_cp_a2a_alltoall(po, ss, workspace, 0, cp_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
