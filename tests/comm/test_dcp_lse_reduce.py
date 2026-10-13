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
    recv_o = torch.where(torch.isfinite(partial_lse).unsqueeze(-1), partial_o, 0)
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
    assert decode_cp_a2a_lse_reduce_workspace_size(1, 1, 2, 8, torch.float16) == 112
    payload_offset = (16 + 2 * 4 * 4 + 15) // 16 * 16
    expected = payload_offset + 2 * 4 * 8 * 2 * (128 * 2 + 4)
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
    partial_o[0, 0, :] = float("nan")
    if cp_rank == 0:
        partial_lse[1, 0, :] = float("nan")
        partial_o[1, 0, :] = float("nan")
    if cp_rank == min(1, cp_size - 1):
        partial_lse[1, 1, :] = float("inf")
        partial_o[1, 1, :] = float("nan")

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

    graph.reset()


def _attention_output_case(batch, local_heads, head_dim, dtype, seed):
    """Padded rank-major O and head-major LSE, viewed without copies for FI."""
    cp_rank, cp_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gen = torch.Generator(device=device).manual_seed(seed * 131 + cp_rank)
    heads = cp_size * local_heads
    # Attention producers can retain padded O heads and head-major LSE storage.
    out = torch.randn(
        batch, heads + 8, head_dim, dtype=dtype, device=device, generator=gen
    )[:, :heads]
    lse = 4 * torch.randn(
        heads, batch, dtype=torch.float32, device=device, generator=gen
    ).transpose(0, 1)
    partial_o = out.unflatten(1, (cp_size, local_heads)).permute(0, 2, 1, 3)
    partial_lse = lse.unflatten(1, (cp_size, local_heads)).permute(0, 2, 1)
    return partial_o, partial_lse, (out, lse)


def _workspace(max_tokens, local_heads, head_dim, dtype):
    return decode_cp_a2a_lse_reduce_create_workspace(
        max_tokens=max_tokens,
        local_heads=local_heads,
        cp_size=dist.get_world_size(),
        head_dim=head_dim,
        dtype=dtype,
        group=dist.group.WORLD,
    )


def _reduce(partial_o, partial_lse, workspace, lse_mode="base2", *, mask=None):
    return decode_cp_a2a_lse_reduce(
        partial_o,
        partial_lse,
        workspace,
        cp_rank=dist.get_rank(),
        cp_size=dist.get_world_size(),
        lse_mode=lse_mode,
        mask=mask,
    )


def _distributed_reference(out, lse, lse_mode, mask=None):
    """Select this rank's original producer heads independently of FI views."""
    cp_size, cp_rank = dist.get_world_size(), dist.get_rank()
    heads = out.shape[1] // cp_size
    if mask is not None:
        lse = lse.masked_fill(~mask[:, None], float("-inf"))
    all_o = [torch.empty_like(out.contiguous()) for _ in range(cp_size)]
    all_lse = [torch.empty_like(lse.contiguous()) for _ in range(cp_size)]
    dist.all_gather(all_o, out.contiguous())
    dist.all_gather(all_lse, lse.contiguous())
    mine = slice(cp_rank * heads, (cp_rank + 1) * heads)
    recv_o = torch.stack([o[:, mine] for o in all_o], dim=-2)
    recv_lse = torch.stack([lse[:, mine] for lse in all_lse], dim=-1)
    return _reference_lse_reduce(recv_o, recv_lse, lse_mode)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lse_mode", ["base2", "basee"])
@pytest.mark.parametrize("batch", [6, 8])
def test_lse_reduce_mask(process_group, dtype, lse_mode, batch):
    """Masked or nonfinite-LSE rows exclude NaN O; finite-LSE NaN O stays visible."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    heads, dim = 2, 512
    partial_o, partial_lse, source = _attention_output_case(
        batch, heads, dim, dtype, seed=700
    )
    mask = torch.ones(batch, dtype=torch.bool, device=partial_o.device)
    mask[0] = False
    mask[1] = rank != 0
    mask[5:] = False  # Padding may contain stale finite LSE and undefined O.
    # False mask overrides a stale finite LSE, even when O is undefined.
    partial_o.masked_fill_(~mask[:, None, None, None], float("nan"))
    partial_lse.masked_fill_(~mask[:, None, None], 100.0)
    partial_o[2] = float("nan")
    partial_lse[2] = (float("-inf"), float("inf"), float("nan"))[rank % 3]
    if rank == 0:
        partial_o[3, 0] = float("nan")
        partial_lse[3, 0] = float("nan")
        # Finite-LSE NaN output remains visible, rather than being sanitized.
        partial_o[4, 0] = float("nan")
        partial_lse[4, 0] = 100.0
    expected = _distributed_reference(*source, lse_mode, mask)
    ws = _workspace(batch, heads, dim, dtype)
    actual = _reduce(partial_o, partial_lse, ws, lse_mode, mask=mask)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3, equal_nan=True)
    assert torch.equal(actual[0], torch.zeros_like(actual[0]))
    assert torch.equal(actual[2], torch.zeros_like(actual[2]))
    assert torch.isnan(actual[4, 0]).all()
    assert torch.equal(actual[5:], torch.zeros_like(actual[5:]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lse_mode", ["base2", "basee"])
@pytest.mark.parametrize("graph_calls", [1, 2])
def test_lse_reduce_epoch_reuse(process_group, dtype, lse_mode, graph_calls):
    """Changing inputs, shapes, and graph call counts share one device epoch."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, cp_size = dist.get_rank(), dist.get_world_size()
    heads, dim = 2, 512
    batches = (1, 4, 8, 4, 1, 4, 8, 4, 1, 8)
    cases = []
    for step, batch in enumerate(batches):
        partial_o, partial_lse, source = _attention_output_case(
            batch, heads, dim, dtype, seed=800 + step
        )
        mask = torch.ones(batch, dtype=torch.bool, device=partial_o.device)
        mask[0] = (rank + step) % 3 != 0
        mask[5:] = False  # Graph padding may retain finite LSE and NaN O.
        if step in (4, 6):
            mask.zero_()
        if batch > 1:
            partial_lse[2] = float("-inf")
            partial_o[2] = float("nan")
        partial_o.masked_fill_(~mask[:, None, None, None], float("nan"))
        expected = _distributed_reference(*source, lse_mode, mask)
        cases.append((partial_o, partial_lse, mask, expected))

    # Finish reference collectives before testing slot reuse under rank skew.
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graphs = {}
    outputs = []
    try:
        with torch.cuda.stream(stream):
            ws = _workspace(max(batches), heads, dim, dtype)
            for batch in (1, 4, 8):
                partial_o, partial_lse, mask, _ = cases[batches.index(batch)]
                _reduce(partial_o, partial_lse, ws, lse_mode, mask=mask)
            stream.synchronize()
            dist.barrier()
            # Shapes on either side of the row-distributed threshold share ws.
            for batch in (1, 8):
                partial_o, partial_lse, mask, _ = cases[batches.index(batch)]
                graph = torch.cuda.CUDAGraph()
                graph_outputs = []
                graphs[batch] = (graph, partial_o, partial_lse, mask, graph_outputs)
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(graph_calls):
                        graph_outputs.append(
                            _reduce(partial_o, partial_lse, ws, lse_mode, mask=mask)
                        )

            for step, (batch, case) in enumerate(zip(batches, cases, strict=True)):
                partial_o, partial_lse, mask, expected = case
                if rank == step % cp_size:
                    torch.cuda._sleep(2_000_000)
                if batch in graphs:
                    graph, static_o, static_lse, static_mask, graph_outputs = graphs[
                        batch
                    ]
                    static_o.copy_(partial_o)
                    static_lse.copy_(partial_lse)
                    static_mask.copy_(mask)
                    graph.replay()
                    # Preserve every invocation before a later replay overwrites it.
                    outputs.extend((out.clone(), expected) for out in graph_outputs)
                else:
                    actual = _reduce(partial_o, partial_lse, ws, lse_mode, mask=mask)
                    outputs.append((actual, expected))

        # No per-call synchronization or reference collective hides stale slots.
        stream.synchronize()
        for actual, expected in outputs:
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
            if torch.count_nonzero(expected) == 0:
                assert torch.count_nonzero(actual) == 0
    finally:
        for graph, *_ in graphs.values():
            graph.reset()


@pytest.mark.parametrize(
    "invalid", ["dtype", "device", "length", "ndim", "stride", "input_ndim"]
)
def test_mask_validation(process_group, invalid):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    partial_o, partial_lse, _ = _attention_output_case(
        4, 2, 64, torch.bfloat16, seed=710
    )
    mask = torch.ones(4, dtype=torch.bool, device=partial_o.device)
    if invalid == "dtype":
        mask = mask.float()
    elif invalid == "device":
        mask = mask.cpu()
    elif invalid == "length":
        mask = mask[:3]
    elif invalid == "ndim":
        mask = mask[:, None]
    elif invalid == "stride":
        mask = torch.ones(8, dtype=torch.bool, device=partial_o.device)[::2]
    elif invalid == "input_ndim":
        partial_o = partial_o.contiguous().flatten(0, 1)
        partial_lse = partial_lse.contiguous().flatten(0, 1)
    ws = _workspace(4, 2, 64, torch.bfloat16)
    with pytest.raises(RuntimeError, match="mask"):
        _reduce(partial_o, partial_lse, ws, mask=mask)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lse_mode", ["base2", "basee"])
def test_lse_reduce_strided_input(process_group, dtype, lse_mode):
    local_heads, head_dim = 2, 512
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ws = _workspace(8, local_heads, head_dim, dtype)
    for batch in (1, 3, 8):
        partial_o, partial_lse, source = _attention_output_case(
            batch, local_heads, head_dim, dtype, seed=batch
        )
        expected = _distributed_reference(*source, lse_mode)
        strided = _reduce(partial_o, partial_lse, ws, lse_mode)
        packed = _reduce(partial_o.contiguous(), partial_lse.contiguous(), ws, lse_mode)
        torch.testing.assert_close(strided, expected, rtol=1e-2, atol=2e-3)
        # The input layout must not change the result.
        assert torch.equal(strided, packed)
    partial_o, partial_lse, _ = _attention_output_case(
        9, local_heads, head_dim, dtype, seed=9
    )
    with pytest.raises(RuntimeError, match="exceeds workspace capacity"):
        _reduce(partial_o, partial_lse, ws, lse_mode)


def test_lse_reduce_contiguous_nd_and_broadcast(process_group):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    partial_o, partial_lse, source = _attention_output_case(
        4, 2, 64, torch.bfloat16, seed=720
    )
    expected = _distributed_reference(*source, "base2")
    ws = _workspace(4, 2, 64, torch.bfloat16)
    n = dist.get_world_size()
    # None preserves arbitrary contiguous leading dimensions and output shape.
    for leading in ((8,), (2, 2, 2)):
        output = _reduce(
            partial_o.contiguous().reshape(*leading, n, 64),
            partial_lse.contiguous().reshape(*leading, n),
            ws,
        )
        torch.testing.assert_close(
            output.reshape_as(expected), expected, rtol=1e-2, atol=2e-3
        )
    # Outer stride zero repeats read-only source rows; workspace rows stay independent.
    partial_o = partial_o[:1, :1].expand(4, 2, n, 64)
    partial_lse = partial_lse[:1, :1, :1].expand(4, 2, n)
    mask = torch.ones(4, dtype=torch.bool, device=partial_o.device)
    expected = _distributed_reference(
        partial_o.transpose(1, 2).flatten(1, 2),
        partial_lse.transpose(1, 2).flatten(1, 2),
        "base2",
        mask,
    )
    actual = _reduce(partial_o, partial_lse, ws, mask=mask)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("invalid", ["pointer", "row_stride", "inner_stride", "ndim"])
def test_lse_reduce_stride_validation(process_group, invalid):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    n = dist.get_world_size()
    shape = (4, 2, n, 64)
    partial_o = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    partial_lse = torch.zeros(shape[:-1], dtype=torch.float32, device=device)
    if invalid == "pointer":
        partial_o = torch.zeros(
            partial_o.numel() + 1, dtype=partial_o.dtype, device=device
        )[1:].view(shape)
    elif invalid == "row_stride":
        partial_o = torch.zeros(4, 2, n, 65, dtype=partial_o.dtype, device=device)[
            ..., :64
        ]
    elif invalid == "inner_stride":
        partial_o = torch.zeros(4, 2, n, 128, dtype=partial_o.dtype, device=device)[
            ..., ::2
        ]
    else:
        partial_o = partial_o.transpose(0, 1).unsqueeze(0)
        partial_lse = partial_lse.transpose(0, 1).unsqueeze(0)
    ws = _workspace(4, 2, 64, torch.bfloat16)
    with pytest.raises(RuntimeError, match="16-byte|stride|strided inputs"):
        _reduce(partial_o, partial_lse, ws)
