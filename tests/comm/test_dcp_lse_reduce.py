# Copyright (c) 2026 by FlashInfer team.
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

"""Multi-rank tests for flashinfer.comm.decode_cp_a2a_lse_reduce.

Run with one process per GPU:

  torchrun --standalone --nproc-per-node=4 \
    -m pytest tests/comm/test_dcp_lse_reduce.py -v -s
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from flashinfer.comm import (
    decode_cp_a2a_lse_reduce,
    decode_cp_a2a_lse_reduce_create_workspace,
    decode_cp_a2a_lse_reduce_workspace_size,
)


def _backend_available() -> bool:
    if not torch.cuda.is_available() or "RANK" not in os.environ:
        return False
    try:
        symm_mem.set_backend("NCCL")
        return symm_mem.get_backend(torch.device("cuda")) == "NCCL"
    except (RuntimeError, AttributeError):
        return False


pytestmark = pytest.mark.skipif(
    not _backend_available(),
    reason="Requires torchrun, CUDA, and torch's NCCL symmetric-memory backend",
)


@pytest.fixture(scope="module", autouse=True)
def process_group():
    created = False
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        created = True
    yield
    if created:
        dist.destroy_process_group()


def _reference_lse_reduce(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    is_lse_base_on_e: bool,
) -> torch.Tensor:
    recv_o = partial_o
    recv_lse = partial_lse.clone()
    recv_lse = torch.where(
        torch.isnan(recv_lse) | torch.isposinf(recv_lse),
        torch.full_like(recv_lse, float("-inf")),
        recv_lse,
    )
    lse_max = recv_lse.max(dim=-1, keepdim=True).values
    lse_max = torch.where(torch.isneginf(lse_max), torch.zeros_like(lse_max), lse_max)
    weights = (
        torch.exp(recv_lse - lse_max)
        if is_lse_base_on_e
        else torch.exp2(recv_lse - lse_max)
    )
    denom = weights.sum(dim=-1, keepdim=True)
    expected = (
        (recv_o.float() * weights.unsqueeze(-1)).sum(dim=-2) / denom.clamp_min(1e-20)
    )
    expected = torch.where(denom == 0, torch.zeros_like(expected), expected)
    return expected.to(partial_o.dtype)


def test_workspace_size():
    expected = 16 + 2 * 4 * 8 * 2 * (128 * 2 + 4)
    assert (
        decode_cp_a2a_lse_reduce_workspace_size(
            max_tokens=8,
            local_heads=2,
            cp_size=4,
            head_dim=128,
            dtype=torch.bfloat16,
        )
        == expected
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
def test_lse_reduce(dtype, is_lse_base_on_e):
    torch.manual_seed(0)
    group = dist.group.WORLD
    cp_rank = dist.get_rank(group)
    cp_size = dist.get_world_size(group)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    batch, local_heads, head_dim = 4, 2, 64
    device = torch.device("cuda", local_rank)
    partial_o = torch.randn(
        batch, local_heads, cp_size, head_dim, dtype=dtype, device=device
    )
    partial_lse = torch.randn(
        batch, local_heads, cp_size, dtype=torch.float32, device=device
    )
    # One globally empty head: output row must be zeros after sanitise.
    partial_lse[0, 0, :] = float("-inf")
    if cp_rank == 0:
        partial_lse[1, 0, :] = float("nan")
    if cp_rank == min(1, cp_size - 1):
        partial_lse[1, 1, :] = float("inf")

    all_o = [torch.empty_like(partial_o) for _ in range(cp_size)]
    all_lse = [torch.empty_like(partial_lse) for _ in range(cp_size)]
    dist.all_gather(all_o, partial_o, group=group)
    dist.all_gather(all_lse, partial_lse, group=group)

    ws = decode_cp_a2a_lse_reduce_create_workspace(
        max_tokens=batch + 1,
        local_heads=local_heads,
        cp_size=cp_size,
        head_dim=head_dim,
        dtype=dtype,
        group=group,
    )
    # Three calls exercise slot 0, slot 1, and slot 0 reuse without re-init.
    for _ in range(3):
        actual = decode_cp_a2a_lse_reduce(
            partial_o,
            partial_lse,
            ws,
            cp_rank=cp_rank,
            cp_size=cp_size,
            is_lse_base_on_e=is_lse_base_on_e,
            enable_pdl=None,
        )

    recv_o = torch.stack([tensor[..., cp_rank, :] for tensor in all_o], dim=-2)
    recv_lse = torch.stack([tensor[..., cp_rank] for tensor in all_lse], dim=-1)
    expected = _reference_lse_reduce(recv_o, recv_lse, is_lse_base_on_e)
    assert actual.shape == (batch, local_heads, head_dim)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)

    # Capture one invocation on every rank, then replay collectively.
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = decode_cp_a2a_lse_reduce(
            partial_o,
            partial_lse,
            ws,
            cp_rank=cp_rank,
            cp_size=cp_size,
            is_lse_base_on_e=is_lse_base_on_e,
        )
    dist.barrier(group=group)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, expected, rtol=1e-2, atol=1e-3)
