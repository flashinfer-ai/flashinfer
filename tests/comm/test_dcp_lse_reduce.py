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


@pytest.fixture(scope="module")
def process_group():
    if not _backend_available():
        pytest.skip(
            "Requires torchrun, CUDA, and torch's NCCL symmetric-memory backend"
        )
    created = False
    if not dist.is_initialized():
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
        # Symmetric memory needs the process group's NCCL host communicator,
        # which is created by the first eager collective in released PyTorch.
        warmup = torch.zeros(1, device=device)
        dist.all_reduce(warmup)
        created = True
    yield
    if created:
        dist.destroy_process_group()


def _reference_lse_reduce(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    lse_mode: str,
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
        if lse_mode == "basee"
        else torch.exp2(recv_lse - lse_max)
    )
    denom = weights.sum(dim=-1, keepdim=True)
    expected = (recv_o.float() * weights.unsqueeze(-1)).sum(dim=-2) / denom.clamp_min(
        1e-20
    )
    expected = torch.where(denom == 0, torch.zeros_like(expected), expected)
    return expected.to(partial_o.dtype)


def test_workspace_size():
    # 128 bytes of metadata, then per (slot, source, row): 128 bf16 values are 32
    # eight-byte words -> 3 lines of 15 words (128 bytes each) + 1 LSE line of 16.
    expected = 128 + 2 * 4 * 8 * 2 * (3 * 128 + 16)
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


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"max_tokens": 0}, ValueError),
        ({"local_heads": 0}, ValueError),
        ({"cp_size": 65}, ValueError),
        ({"head_dim": 0}, ValueError),
        ({"head_dim": 7}, ValueError),
        ({"dtype": torch.float32}, TypeError),
    ],
)
def test_workspace_size_validation(kwargs, error):
    params = {
        "max_tokens": 8,
        "local_heads": 2,
        "cp_size": 4,
        "head_dim": 128,
        "dtype": torch.bfloat16,
    }
    params.update(kwargs)
    with pytest.raises(error):
        decode_cp_a2a_lse_reduce_workspace_size(**params)


def test_lse_mode_validation():
    with pytest.raises(ValueError, match="lse_mode must be 'base2' or 'basee'"):
        decode_cp_a2a_lse_reduce(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            cp_rank=0,
            cp_size=1,
            lse_mode="invalid",
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lse_mode", ["base2", "basee"])
def test_single_rank_reference_is_identity(dtype, lse_mode):
    partial_o = torch.randn(3, 1, 16, dtype=dtype)
    partial_lse = torch.randn(3, 1)
    actual = _reference_lse_reduce(partial_o, partial_lse, lse_mode)
    torch.testing.assert_close(actual, partial_o[:, 0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lse_mode", ["base2", "basee"])
def test_lse_reduce(process_group, dtype, lse_mode):
    group = dist.group.WORLD
    cp_rank = dist.get_rank(group)
    cp_size = dist.get_world_size(group)
    torch.manual_seed(cp_rank)
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
    recv_o = torch.stack([tensor[..., cp_rank, :] for tensor in all_o], dim=-2)
    recv_lse = torch.stack([tensor[..., cp_rank] for tensor in all_lse], dim=-1)
    expected = _reference_lse_reduce(recv_o, recv_lse, lse_mode)

    # Three calls exercise slot 0, slot 1, and slot 0 reuse without re-init.
    for _ in range(3):
        actual = decode_cp_a2a_lse_reduce(
            partial_o,
            partial_lse,
            ws,
            cp_rank=cp_rank,
            cp_size=cp_size,
            lse_mode=lse_mode,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)

    assert actual.shape == (batch, local_heads, head_dim)

    # The workspace uses two shared slots and must stay on one ordered stream.
    other_stream = torch.cuda.Stream(device=device)
    with (
        torch.cuda.stream(other_stream),
        pytest.raises(RuntimeError, match="one ordered CUDA stream"),
    ):
        decode_cp_a2a_lse_reduce(
            partial_o,
            partial_lse,
            ws,
            cp_rank=cp_rank,
            cp_size=cp_size,
            lse_mode=lse_mode,
        )

    # CUDA graph capture uses a distinct stream, so it needs a separately
    # rendezvoused workspace under the one-ordered-stream workspace contract.
    graph_ws = decode_cp_a2a_lse_reduce_create_workspace(
        max_tokens=batch + 1,
        local_heads=local_heads,
        cp_size=cp_size,
        head_dim=head_dim,
        dtype=dtype,
        group=group,
    )
    # Capture one invocation on every rank, then replay enough times to exercise
    # both slots and slot reuse inside a graph.
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = decode_cp_a2a_lse_reduce(
            partial_o,
            partial_lse,
            graph_ws,
            cp_rank=cp_rank,
            cp_size=cp_size,
            lse_mode=lse_mode,
        )
    dist.barrier(group=group)
    for _ in range(4):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_out, expected, rtol=1e-2, atol=1e-3)


def _attention_output_case(
    batch: int,
    local_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    lse_mode: str,
    seed: int,
):
    """This rank's attention output and LSE for ``cp_size * local_heads`` heads,
    viewed as the ``[batch, local_heads, cp_size, ...]`` inputs of the op (head
    ``h`` goes to peer ``h // local_heads``), and this rank's expected result."""
    group = dist.group.WORLD
    cp_rank = dist.get_rank(group)
    cp_size = dist.get_world_size(group)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gen = torch.Generator(device=device).manual_seed(seed * 131 + cp_rank)
    heads = cp_size * local_heads
    out = torch.randn(batch, heads, head_dim, dtype=dtype, device=device, generator=gen)
    lse = 4 * torch.randn(
        batch, heads, dtype=torch.float32, device=device, generator=gen
    )
    all_out = [torch.empty_like(out) for _ in range(cp_size)]
    all_lse = [torch.empty_like(lse) for _ in range(cp_size)]
    dist.all_gather(all_out, out, group=group)
    dist.all_gather(all_lse, lse, group=group)
    mine = slice(cp_rank * local_heads, (cp_rank + 1) * local_heads)
    recv_o = torch.stack([o[:, mine] for o in all_out], dim=-2)
    recv_lse = torch.stack([s[:, mine] for s in all_lse], dim=-1)
    expected = _reference_lse_reduce(recv_o, recv_lse, lse_mode)
    partial_o = out.unflatten(1, (cp_size, local_heads)).permute(0, 2, 1, 3)
    partial_lse = lse.unflatten(1, (cp_size, local_heads)).permute(0, 2, 1)
    return partial_o, partial_lse, expected


def _workspace(max_tokens: int, local_heads: int, head_dim: int, dtype: torch.dtype):
    return decode_cp_a2a_lse_reduce_create_workspace(
        max_tokens=max_tokens,
        local_heads=local_heads,
        cp_size=dist.get_world_size(),
        head_dim=head_dim,
        dtype=dtype,
        group=dist.group.WORLD,
    )


def _reduce(partial_o, partial_lse, workspace, lse_mode="base2"):
    return decode_cp_a2a_lse_reduce(
        partial_o,
        partial_lse,
        workspace,
        cp_rank=dist.get_rank(),
        cp_size=dist.get_world_size(),
        lse_mode=lse_mode,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lse_mode", ["base2", "basee"])
@pytest.mark.parametrize("local_heads,head_dim", [(16, 512), (2, 128), (4, 64)])
def test_lse_reduce_strided_input(
    process_group, dtype, lse_mode, local_heads, head_dim
):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ws = _workspace(8, local_heads, head_dim, dtype)
    for batch in (1, 3, 8):
        partial_o, partial_lse, expected = _attention_output_case(
            batch, local_heads, head_dim, dtype, lse_mode, seed=batch
        )
        strided = _reduce(partial_o, partial_lse, ws, lse_mode)
        packed = _reduce(partial_o.contiguous(), partial_lse.contiguous(), ws, lse_mode)
        torch.testing.assert_close(strided, expected, rtol=1e-2, atol=2e-3)
        # The input layout must not change the result.
        assert torch.equal(strided, packed)


def test_lse_reduce_row_counts(process_group):
    # One workspace serves calls with different row counts, in any order.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    local_heads, head_dim, dtype = 16, 512, torch.bfloat16
    ws = _workspace(32, local_heads, head_dim, dtype)
    for i, batch in enumerate((7, 1, 32, 3, 1, 32, 2)):
        partial_o, partial_lse, expected = _attention_output_case(
            batch, local_heads, head_dim, dtype, "base2", seed=100 + i
        )
        torch.testing.assert_close(
            _reduce(partial_o, partial_lse, ws), expected, rtol=1e-2, atol=2e-3
        )


def _set_call_counter(workspace: torch.Tensor, value: int) -> None:
    # White-box: the first workspace word is the device-side call counter that
    # selects the slot and the flag value.
    torch.cuda.synchronize()
    dist.barrier()
    workspace[:4].view(torch.int32).fill_(
        value - (1 << 32) if value >= 1 << 31 else value
    )
    torch.cuda.synchronize()
    dist.barrier()


def test_lse_reduce_call_counter_reuse_and_wrap(process_group):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    local_heads, head_dim, dtype = 16, 512, torch.bfloat16
    ws = _workspace(8, local_heads, head_dim, dtype)
    # A repeated call number with new data must not return the previous call's lines.
    for i in range(3):
        _set_call_counter(ws, 10)
        partial_o, partial_lse, expected = _attention_output_case(
            8, local_heads, head_dim, dtype, "base2", seed=200 + i
        )
        torch.testing.assert_close(
            _reduce(partial_o, partial_lse, ws), expected, rtol=1e-2, atol=2e-3
        )
    # Calls keep working across the 32-bit wrap of the counter.
    _set_call_counter(ws, (1 << 32) - 3)
    for i, batch in enumerate((8, 1, 5, 8, 2, 8)):
        partial_o, partial_lse, expected = _attention_output_case(
            batch, local_heads, head_dim, dtype, "base2", seed=300 + i
        )
        torch.testing.assert_close(
            _reduce(partial_o, partial_lse, ws), expected, rtol=1e-2, atol=2e-3
        )
    assert int(ws[:4].view(torch.int32).item()) == 3


def test_lse_reduce_graph_many_calls(process_group):
    # Every replay of a graph with many captured calls advances the device-side
    # call counter through both slots.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    local_heads, head_dim, dtype = 16, 512, torch.bfloat16
    cases = [
        _attention_output_case(
            1 + i % 4, local_heads, head_dim, dtype, "base2", seed=400 + i
        )
        for i in range(16)
    ]
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ws = _workspace(4, local_heads, head_dim, dtype)
        # An eager call binds the workspace to the capture stream.
        for partial_o, partial_lse, expected in cases:
            torch.testing.assert_close(
                _reduce(partial_o, partial_lse, ws), expected, rtol=1e-2, atol=2e-3
            )
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outs = [_reduce(o, lse, ws) for o, lse, _ in cases]
    dist.barrier()
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    for (_, _, expected), out in zip(cases, outs, strict=True):
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=2e-3)
