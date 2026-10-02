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

Correctness and protocol contracts for grouped BF16 combine.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import pytest
import torch


_PACKAGE_NAME = "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe"
_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "flashinfer"
    / "moe_ep"
    / "kernel_src"
    / "sm90"
    / "push_style_megamoe"
)


def _package_text(*parts: str) -> str:
    source_tree = _PACKAGE_ROOT.joinpath(*parts)
    if source_tree.is_file():
        return source_tree.read_text(encoding="utf-8")

    resource = resources.files(_PACKAGE_NAME)
    for part in parts:
        resource = resource / part
    return resource.read_text(encoding="utf-8")


def _sm90_cuda_12_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.jit.cpp_ext import is_cuda_version_at_least
        from flashinfer.utils import is_sm90a_supported

        return is_cuda_version_at_least("12.0") and is_sm90a_supported(
            torch.device("cuda")
        )
    except Exception:
        return False


_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
requires_sm90 = pytest.mark.skipif(
    not _sm90_cuda_12_available() or _WORLD > 1,
    reason="requires one SM90 GPU and CUDA Toolkit 12.0+ outside torchrun",
)
requires_dist = pytest.mark.skipif(
    _WORLD < 2 or not _sm90_cuda_12_available(),
    reason="requires torchrun with at least two SM90 GPUs and CUDA Toolkit 12.0+",
)


def _align(value: int, alignment: int = 128) -> int:
    return (value + alignment - 1) // alignment * alignment


def _grouped_bf16_reference(
    partials: torch.Tensor,
    meta: torch.Tensor,
    row_map: torch.Tensor,
    *,
    ep_size: int,
    num_tokens: int,
) -> torch.Tensor:
    hidden = partials.shape[1]
    inbox = torch.zeros(
        num_tokens,
        ep_size,
        hidden,
        dtype=torch.bfloat16,
        device=partials.device,
    )
    groups: dict[tuple[int, int], list[int]] = {}
    for logical_row in range(meta.shape[0]):
        source_rank = int(meta[logical_row, 0])
        source_token = int(meta[logical_row, 1])
        groups.setdefault((source_rank, source_token), []).append(logical_row)
    for (source_rank, source_token), rows in groups.items():
        rows.sort(key=lambda row: int(meta[row, 2]))
        acc = torch.zeros(hidden, dtype=torch.float32, device=partials.device)
        for logical_row in rows:
            mapped_row = int(row_map[logical_row])
            weight = meta[logical_row, 3].view(torch.float32)
            acc.add_(partials[mapped_row].float() * weight)
        inbox[source_token, source_rank].copy_(acc)
    return inbox.float().sum(dim=1)


@pytest.mark.parametrize("top_k", [1, 2, 8])
def test_grouped_bf16_cpu_oracle_handles_route_and_mapping_boundaries(
    top_k: int,
) -> None:
    hidden = 16
    rows = 2 * top_k
    partials = torch.arange(rows * hidden, dtype=torch.float32).reshape(rows, hidden)
    partials = (partials / 37.0).to(torch.bfloat16)
    row_map = torch.arange(rows - 1, -1, -1, dtype=torch.int32)
    meta = torch.empty(rows, 4, dtype=torch.int32)
    for row in range(rows):
        meta[row, 0] = row & 1
        meta[row, 1] = 0
        meta[row, 2] = row // 2
        weight = torch.tensor((row + 1) / (rows + 1), dtype=torch.float32)
        meta[row, 3] = weight.view(torch.int32)

    actual = _grouped_bf16_reference(
        partials,
        meta,
        row_map,
        ep_size=2,
        num_tokens=1,
    )

    expected = torch.zeros(hidden, dtype=torch.float32)
    for source_rank in range(2):
        local = torch.zeros(hidden, dtype=torch.float32)
        for route in range(top_k):
            row = route * 2 + source_rank
            weight = meta[row, 3].view(torch.float32)
            local.add_(partials[int(row_map[row])].float() * weight)
        expected.add_(local.to(torch.bfloat16).float())
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)


def test_grouped_bf16_cpu_oracle_keeps_source_rounding_boundary() -> None:
    partials = torch.tensor([[0.3333], [0.3333]], dtype=torch.bfloat16)
    row_map = torch.tensor([0, 1], dtype=torch.int32)
    meta = torch.empty(2, 4, dtype=torch.int32)
    meta[:, 0] = torch.tensor([0, 1], dtype=torch.int32)
    meta[:, 1] = 0
    meta[:, 2] = 0
    meta[:, 3] = torch.tensor(1.0, dtype=torch.float32).view(torch.int32)

    actual = _grouped_bf16_reference(
        partials,
        meta,
        row_map,
        ep_size=2,
        num_tokens=1,
    )
    expected = partials[0].float() + partials[1].float()
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)


def test_grouped_bf16_source_contracts_and_fp8_exports() -> None:
    ops = _package_text("src", "a2a", "sm90_push_a2a_ops.cu")
    header = _package_text("src", "a2a", "sm90_push_a2a.cuh")
    protocol = _package_text("shim", "protocol.py")

    assert "combine_row_grouped(int r, int token, int src)" in header
    assert "proto_combine_bf16_grouped_mapped" in protocol
    assert "proto_reduce_bf16_grouped" in protocol
    assert "token_capacity * cslots * hidden_size * 2" in protocol
    assert "combine_group_build_kernel<<<" in ops
    assert "combine_publish_bf16_grouped_mapped_kernel<<<" in ops
    assert "L.combine_row_grouped(dst, tok, L.rank)" in ops
    assert "pack_count_tag(groups_per_src[dst], tag)" in ops
    assert "publish_abort_all(L, tag)" in ops
    for export in (
        "sm90_push_combine_fp8",
        "sm90_push_combine_fp8_grouped",
        "sm90_combine_reduce_fp8",
        "sm90_combine_reduce_fp8_grouped",
    ):
        assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export}, {export})" in ops


def test_grouped_bf16_proto_is_fail_closed() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        Sm90PushCombine,
        Sm90PushConfig,
        Sm90PushPipe,
    )

    pipe = object.__new__(Sm90PushPipe)
    pipe.config = Sm90PushConfig(
        combine_dtype=Sm90PushCombine.BF16,
        grouped_combine=True,
    )
    with pytest.raises(RuntimeError, match="mapped grouped interface"):
        pipe.proto_combine(None, None)
    with pytest.raises(RuntimeError, match="proto_combine_bf16_grouped_mapped"):
        pipe.proto_combine_mapped(None, None, None)

    pipe.config = Sm90PushConfig(combine_dtype=Sm90PushCombine.BF16)
    with pytest.raises(RuntimeError, match="grouped_combine=True"):
        pipe.proto_combine_bf16_grouped_mapped(None, None, None)
    with pytest.raises(RuntimeError, match="grouped_combine=True"):
        pipe.proto_reduce_bf16_grouped(None, 0)


def test_grouped_bf16_window_footprint_uses_source_slots() -> None:
    token_capacity = 17
    top_k = 8
    hidden = 256
    for ep_size in (1, 2, 4, 8, 32):
        grouped_bytes = token_capacity * ep_size * hidden * 2
        route_bytes = token_capacity * top_k * hidden * 2
        if ep_size <= top_k:
            assert grouped_bytes <= route_bytes
        else:
            assert grouped_bytes > route_bytes


def _build_pipe(
    *,
    rank: int,
    world: int,
    device: torch.device,
    top_k: int,
    token_capacity: int,
    local_experts: int,
    hidden: int,
    comm_backend=None,
):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        Sm90PushCombine,
        Sm90PushConfig,
        Sm90PushPayload,
        Sm90PushPipe,
    )

    return Sm90PushPipe(
        ep_size=world,
        rank=rank,
        num_local_experts=local_experts,
        hidden_size=hidden,
        top_k=top_k,
        token_capacity=token_capacity,
        device_index=device.index,
        config=Sm90PushConfig(
            payload_dtype=Sm90PushPayload.BF16,
            combine_dtype=Sm90PushCombine.BF16,
            fuse_act=False,
            grouped_combine=True,
        ),
        comm_backend=comm_backend,
    )


def _run_grouped_identity(
    pipe,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    padded_rows = _align(pipe.m_cap + 7 * pipe.E)
    logical_rows = _align(pipe.m_cap)
    a = torch.empty(padded_rows, pipe.H, dtype=torch.bfloat16, device=pipe.device)
    meta = torch.empty(logical_rows, 4, dtype=torch.int32, device=pipe.device)
    row_map = torch.empty(logical_rows, dtype=torch.int32, device=pipe.device)
    offsets = torch.empty(pipe.E + 1, dtype=torch.int64, device=pipe.device)
    tile_prefix = torch.empty_like(offsets)
    padded_m = torch.zeros(1, dtype=torch.int32, device=pipe.device)
    output = torch.empty(x.shape[0], pipe.H, dtype=torch.float32, device=pipe.device)

    pipe.proto_begin_round()
    try:
        pipe.proto_dispatch(x, topk_ids, topk_weights)
        pipe.proto_wait_prefix()
        pipe.proto_compact_bf16_padded(
            a,
            meta,
            row_map,
            offsets,
            tile_prefix,
            padded_m,
            128,
        )
        pipe.proto_combine_bf16_grouped_mapped(a, meta, row_map)
        pipe.proto_wait_combine()
        pipe.proto_reduce_bf16_grouped(output, x.shape[0])
        pipe.proto_ack()
    except Exception:
        if pipe._round_open:
            pipe.proto_abort()
        raise
    return output


def _routing(
    *,
    rank: int,
    world: int,
    num_tokens: int,
    top_k: int,
    local_experts: int,
    device: torch.device,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_experts = world * local_experts
    token = torch.arange(num_tokens, dtype=torch.int32).unsqueeze(1)
    route = torch.arange(top_k, dtype=torch.int32).unsqueeze(0)
    ids = (token * top_k + route + rank) % total_experts
    if mode == "all_remote" and world > 1:
        remote_rank = (rank + 1) % world
        ids = remote_rank * local_experts + ids % local_experts
    elif mode == "hot":
        ids.zero_()
    weights = torch.arange(1, top_k + 1, dtype=torch.float32).expand(num_tokens, -1)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return ids.to(device), weights.to(device)


@requires_sm90
@pytest.mark.parametrize(("num_tokens", "top_k"), [(1, 1), (7, 2), (16, 8)])
def test_ep1_grouped_bf16_identity_numerics(num_tokens: int, top_k: int) -> None:
    device = torch.device("cuda", 0)
    pipe = _build_pipe(
        rank=0,
        world=1,
        device=device,
        top_k=top_k,
        token_capacity=16,
        local_experts=8,
        hidden=128,
    )
    generator = torch.Generator(device="cpu").manual_seed(71 + top_k)
    x = torch.randn(num_tokens, 128, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    ids, weights = _routing(
        rank=0,
        world=1,
        num_tokens=num_tokens,
        top_k=top_k,
        local_experts=8,
        device=device,
        mode="hot" if top_k == 8 else "balanced",
    )
    output = _run_grouped_identity(pipe, x, ids, weights)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(output, x.float(), rtol=3e-3, atol=3e-3)
    assert pipe.combine_t.shape == (16, 1, 128)
    pipe.destroy()


def _dist_setup():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", str(rank))))
    from flashinfer.comm.mnnvl import TorchDistBackend

    return rank, world, TorchDistBackend(group=dist.group.WORLD)


@requires_dist
@pytest.mark.parametrize("mode", ["balanced", "all_remote", "hot", "uneven"])
def test_multirank_grouped_bf16_identity_numerics(mode: str) -> None:
    import torch.distributed as dist

    rank, world, comm = _dist_setup()
    device = torch.device("cuda", rank)
    top_k = min(4, world * 4)
    num_tokens = 0 if mode == "uneven" and rank == world - 1 else 13
    pipe = _build_pipe(
        rank=rank,
        world=world,
        device=device,
        top_k=top_k,
        token_capacity=16,
        local_experts=4,
        hidden=128,
        comm_backend=comm,
    )
    generator = torch.Generator(device="cpu").manual_seed(113 + rank)
    x = torch.randn(num_tokens, 128, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    ids, weights = _routing(
        rank=rank,
        world=world,
        num_tokens=num_tokens,
        top_k=top_k,
        local_experts=4,
        device=device,
        mode="balanced" if mode == "uneven" else mode,
    )
    output = _run_grouped_identity(pipe, x, ids, weights)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(output, x.float(), rtol=3e-3, atol=3e-3)
    assert pipe.combine_t.shape == (16, world, 128)
    dist.barrier()
    pipe.destroy()
