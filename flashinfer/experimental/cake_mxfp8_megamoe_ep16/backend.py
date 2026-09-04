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

# Exact-SM103a Cake MXFP8 MegaMoE backend for EP16.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from ...moe_ep.cake_mxfp8_megamoe_ep16 import CakeMxfp8MegaMoeEp16Weights
from ...comm.torch_symmetric_memory import _enable_symm_mem_for_group
from .jit import (
    load_cake_mxfp8_megamoe_ep16_module,
)

_WORLD_SIZE = 16
_LOCAL_EXPERTS = 32
_EXPERTS = _WORLD_SIZE * _LOCAL_EXPERTS
_TOP_K = 8
_HIDDEN = 3072
_INTERMEDIATE = 5120
_FC1_ROWS = 2 * _INTERMEDIATE
_MAX_TOKENS_PER_RANK = 64
_MAX_RECV_TOKENS = _WORLD_SIZE * _MAX_TOKENS_PER_RANK
_MAX_ROUTES = _MAX_RECV_TOKENS * _TOP_K
_ROWS_PER_EXPERT = 128
_TOTAL_EXPERT_ROWS = _LOCAL_EXPERTS * _ROWS_PER_EXPERT
_THREADS = 256
_SUPPORTED_TOKENS = (16, 32, 64)
_ROUTE_GROUPS = _LOCAL_EXPERTS // _TOP_K


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _interleave_gate_up_128(w13: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(w13)
    result_blocks = result.view(
        _LOCAL_EXPERTS,
        _INTERMEDIATE // 128,
        2,
        128,
        _HIDDEN,
    )
    gate = w13[:, :_INTERMEDIATE].view(
        _LOCAL_EXPERTS, _INTERMEDIATE // 128, 128, _HIDDEN
    )
    up = w13[:, _INTERMEDIATE:].view(_LOCAL_EXPERTS, _INTERMEDIATE // 128, 128, _HIDDEN)
    result_blocks[:, :, 0].copy_(up)
    result_blocks[:, :, 1].copy_(gate)
    return result


def _quantize_mxfp8_block32(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    experts, rows, columns = weight.shape
    if columns % 32:
        raise ValueError("MXFP8 reduction dimension must be divisible by 32")
    quantized = torch.empty(
        weight.shape, dtype=torch.float8_e4m3fn, device=weight.device
    )
    scale_codes = torch.empty(
        (experts, rows, columns // 32), dtype=torch.uint8, device=weight.device
    )
    for row_begin in range(0, rows, 128):
        row_end = min(row_begin + 128, rows)
        values = (
            weight[:, row_begin:row_end]
            .float()
            .reshape(experts, row_end - row_begin, columns // 32, 32)
        )
        block_max = values.abs().amax(dim=-1)
        scale_seed = torch.clamp(block_max, min=1.0e-7) / 448.0
        bits = scale_seed.contiguous().view(torch.int32)
        codes = ((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)
        codes.clamp_(1, 254)
        decoded = torch.exp2(codes.float() - 127.0)
        block_quantized = (values / decoded.unsqueeze(-1)).to(torch.float8_e4m3fn)
        quantized[:, row_begin:row_end].copy_(
            block_quantized.reshape(experts, row_end - row_begin, columns)
        )
        scale_codes[:, row_begin:row_end].copy_(codes.to(torch.uint8))
    return quantized, scale_codes


def _pack_scale_n256_k128(scales: torch.Tensor) -> torch.Tensor:
    experts, rows, block_columns = scales.shape
    columns = block_columns * 32
    if rows % 256 or columns % 128:
        raise ValueError("N256/K128 scale packing requires aligned dimensions")
    return (
        scales.view(
            experts,
            rows // 256,
            2,
            4,
            32,
            columns // 128,
            4,
        )
        .permute(0, 1, 5, 2, 4, 3, 6)
        .contiguous()
        .view(-1)
    )


def _pack_scale_n128_k128(scales: torch.Tensor) -> torch.Tensor:
    experts, rows, block_columns = scales.shape
    columns = block_columns * 32
    if rows % 128 or columns % 128:
        raise ValueError("N128/K128 scale packing requires aligned dimensions")
    return (
        scales.view(
            experts,
            rows // 128,
            4,
            32,
            columns // 128,
            4,
        )
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(-1)
    )


def preprocess_cake_mxfp8_megamoe_ep16_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> CakeMxfp8MegaMoeEp16Weights:
    """Quantize canonical rank-local BF16 weights into the Cake layout.

    ``w13`` uses ``[32, 10240, 3072]`` gate-then-up layout and ``w2`` uses
    ``[32, 3072, 5120]``. This setup operation is intentionally outside the
    inference submission path.
    """

    _require_tensor(
        w13,
        name="w13",
        shape=(_LOCAL_EXPERTS, _FC1_ROWS, _HIDDEN),
        dtype=torch.bfloat16,
    )
    _require_tensor(
        w2,
        name="w2",
        shape=(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE),
        dtype=torch.bfloat16,
        device=w13.device,
    )
    interleaved_w13 = _interleave_gate_up_128(w13)
    w13_fp8, w13_scale_plain = _quantize_mxfp8_block32(interleaved_w13)
    w2_fp8, w2_scale_plain = _quantize_mxfp8_block32(w2)
    return CakeMxfp8MegaMoeEp16Weights(
        w13=w13_fp8,
        w13_scale=_pack_scale_n256_k128(w13_scale_plain),
        w2=w2_fp8,
        w2_scale=_pack_scale_n128_k128(w2_scale_plain),
    )


@dataclass
class _SymmetricTensor:
    tensor: torch.Tensor
    handle: Any
    peers: torch.Tensor

    @property
    def local(self) -> torch.Tensor:
        """Return the local symmetric-memory view."""

        return self.tensor


@dataclass(frozen=True)
class _PreparedRouting:
    topk_ids: torch.Tensor
    source_ranks: torch.Tensor
    source_tokens: torch.Tensor
    local_groups: torch.Tensor
    local_rows: torch.Tensor
    work_count: int


def _prepare_balanced_routing(
    topk_ids: torch.Tensor,
    *,
    process_group: dist.ProcessGroup,
    rank: int,
    device: torch.device,
) -> _PreparedRouting:
    tokens = int(topk_ids.shape[0])
    if tokens not in _SUPPORTED_TOKENS:
        raise ValueError(
            f"tokens per rank must be one of {_SUPPORTED_TOKENS}, got {tokens}"
        )
    _require_tensor(
        topk_ids,
        name="topk_ids",
        shape=(tokens, _TOP_K),
        dtype=torch.int64,
        device=device,
    )

    gathered = torch.empty(
        (_WORLD_SIZE * tokens, _TOP_K), dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(gathered, topk_ids, group=process_group)
    first = gathered[:, 0]
    expected = (
        first[:, None] + torch.arange(_TOP_K, dtype=torch.int64, device=device)[None, :]
    )
    valid = bool(
        torch.all((first >= 0) & (first < _EXPERTS) & ((first % _TOP_K) == 0)).item()
        and torch.equal(gathered, expected)
        and torch.all(gathered < _EXPERTS).item()
    )
    if not valid:
        raise ValueError("topk_ids do not satisfy the balanced EP16 route contract")

    owner = first // _LOCAL_EXPERTS
    owned_tokens = torch.nonzero(owner == rank, as_tuple=False).flatten()
    owned_groups = ((first[owned_tokens] - rank * _LOCAL_EXPERTS) // _TOP_K).to(
        torch.int32
    )
    ordered_tokens: list[torch.Tensor] = []
    ordered_groups: list[torch.Tensor] = []
    ordered_rows: list[torch.Tensor] = []
    for local_group in range(_ROUTE_GROUPS):
        group_tokens = owned_tokens[owned_groups == local_group]
        group_count = int(group_tokens.numel())
        if group_count > 32:
            raise ValueError("balanced EP16 route exceeds the 32-row expert capacity")
        ordered_tokens.append(group_tokens)
        ordered_groups.append(
            torch.full((group_count,), local_group, dtype=torch.int32, device=device)
        )
        ordered_rows.append(torch.arange(group_count, dtype=torch.int32, device=device))

    recv_tokens = torch.cat(ordered_tokens)
    work_count = int(recv_tokens.numel())
    if work_count > _MAX_TOKENS_PER_RANK:
        raise ValueError("balanced EP16 owner work exceeds the prepared capacity")
    padding = torch.zeros(
        (_MAX_TOKENS_PER_RANK - work_count,), dtype=torch.int32, device=device
    )

    def padded(values: torch.Tensor) -> torch.Tensor:
        return torch.cat((values.to(torch.int32), padding))

    return _PreparedRouting(
        topk_ids=topk_ids,
        source_ranks=padded(recv_tokens // tokens),
        source_tokens=padded(recv_tokens % tokens),
        local_groups=padded(torch.cat(ordered_groups)),
        local_rows=padded(torch.cat(ordered_rows)),
        work_count=work_count,
    )


def _allocate_symmetric(
    shape: Sequence[int],
    dtype: torch.dtype,
    *,
    device: torch.device,
    group_name: str,
    world_size: int,
) -> _SymmetricTensor:
    tensor = symm_mem.empty(*shape, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    if hasattr(handle, "buffer_ptrs"):
        pointers = [int(handle.buffer_ptrs[rank]) for rank in range(world_size)]
    elif hasattr(handle, "get_buffer"):
        pointers = [
            int(handle.get_buffer(rank, tuple(shape), dtype, 0).data_ptr())
            for rank in range(world_size)
        ]
    else:
        pointers = [
            int(handle.get_remote_tensor(rank, tuple(shape), dtype).data_ptr())
            for rank in range(world_size)
        ]
    if any(pointer == 0 for pointer in pointers):
        raise RuntimeError("symmetric peer mapping is unavailable")
    peers = torch.tensor(pointers, dtype=torch.int64, device=device)
    return _SymmetricTensor(tensor=tensor, handle=handle, peers=peers)


def _aligned_workspace(
    size: int,
    *,
    device: torch.device,
    alignment: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    backing = torch.empty(size + alignment - 1, dtype=torch.uint8, device=device)
    offset = (-int(backing.data_ptr())) % alignment
    workspace = backing[offset : offset + size]
    if int(workspace.data_ptr()) % alignment:
        raise RuntimeError("failed to align TMA descriptor workspace")
    return backing, workspace


class _Workspace:
    def __init__(
        self,
        *,
        device: torch.device,
        group_name: str,
        routing: _PreparedRouting,
    ) -> None:
        self.flags = _allocate_symmetric(
            (2,),
            torch.uint32,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.flags.tensor.zero_()
        self.hidden = _allocate_symmetric(
            (_MAX_TOKENS_PER_RANK, _HIDDEN),
            torch.uint8,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.hidden_scale = _allocate_symmetric(
            (_MAX_TOKENS_PER_RANK, _HIDDEN // 32),
            torch.uint8,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.topk_ids = _allocate_symmetric(
            (_MAX_TOKENS_PER_RANK, _TOP_K),
            torch.int32,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.topk_weights = _allocate_symmetric(
            (_MAX_TOKENS_PER_RANK, _TOP_K),
            torch.float32,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.input_ready = _allocate_symmetric(
            (_MAX_TOKENS_PER_RANK, _WORLD_SIZE),
            torch.uint32,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.input_ready.tensor.zero_()
        self.token_back_ready = _allocate_symmetric(
            (_WORLD_SIZE,),
            torch.uint32,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.token_back_ready.tensor.zero_()
        self.output = _allocate_symmetric(
            (_MAX_TOKENS_PER_RANK, _HIDDEN),
            torch.bfloat16,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )

        self.route_weights = torch.empty(
            _MAX_ROUTES, dtype=torch.float32, device=device
        )
        self.route_map = torch.empty(
            _TOTAL_EXPERT_ROWS, dtype=torch.int32, device=device
        )
        self.fc1_input = torch.empty(
            (_ROUTE_GROUPS, _ROWS_PER_EXPERT, _HIDDEN),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.fc1_input_scale = torch.empty(
            _ROUTE_GROUPS * (_HIDDEN // 128) * 512,
            dtype=torch.uint8,
            device=device,
        )
        self.fc2_input = torch.empty(
            (_LOCAL_EXPERTS, _ROWS_PER_EXPERT, _INTERMEDIATE),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.fc2_input_scale = torch.empty(
            _LOCAL_EXPERTS * (_INTERMEDIATE // 128) * 512,
            dtype=torch.uint8,
            device=device,
        )
        self.fc1_tile_expert = torch.arange(
            _LOCAL_EXPERTS, dtype=torch.int32, device=device
        ).repeat_interleave(_INTERMEDIATE // 128)
        self.fc1_tile_m = torch.arange(
            0, _INTERMEDIATE, 128, dtype=torch.int32, device=device
        ).repeat(_LOCAL_EXPERTS)
        self.fc2_tile_expert = torch.arange(
            _LOCAL_EXPERTS, dtype=torch.int32, device=device
        ).repeat_interleave(_HIDDEN // 128)
        self.fc2_tile_m = torch.arange(
            0, _HIDDEN, 128, dtype=torch.int32, device=device
        ).repeat(_LOCAL_EXPERTS)
        self.expert_row_offsets = torch.arange(
            0, _TOTAL_EXPERT_ROWS, _ROWS_PER_EXPERT, dtype=torch.int32, device=device
        )
        self.ones = torch.ones(_LOCAL_EXPERTS, dtype=torch.float32, device=device)
        self.zeros = torch.zeros(_LOCAL_EXPERTS, dtype=torch.float32, device=device)
        self.infinities = torch.full(
            (_LOCAL_EXPERTS,), float("inf"), dtype=torch.float32, device=device
        )
        self.local_terms = torch.empty(
            (_MAX_ROUTES, _HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        self.fc1_done = torch.zeros((1,), dtype=torch.uint32, device=device)
        self.work_source_ranks = routing.source_ranks
        self.work_source_tokens = routing.source_tokens
        self.work_local_groups = routing.local_groups
        self.work_local_rows = routing.local_rows
        self.work_count = routing.work_count
        self.tma_backing, self.tma_workspace = _aligned_workspace(1024, device=device)

    def destroy(self) -> None:
        """Release resources when the owning session is discarded."""


class CakeMxfp8MegaMoeEp16:
    """Prepared EP16 MXFP8 MegaMoE session for exact SM103a devices.

    The route supports 512 experts, hidden size 3072, intermediate size 5120,
    top-k 8, 16/32/64 tokens per rank, and balanced groups of eight experts.
    Routing is prepared once at construction and must remain immutable.
    Construction owns all symmetric memory and scratch storage.
    :meth:`run` submits on the current CUDA stream without allocating and can
    be captured by a caller-owned CUDA Graph after one warmup call.
    """

    def __init__(
        self,
        weights: CakeMxfp8MegaMoeEp16Weights,
        topk_ids: torch.Tensor,
        *,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        self._group = dist.group.WORLD if process_group is None else process_group
        self.rank = int(dist.get_rank(self._group))
        world_size = int(dist.get_world_size(self._group))
        if world_size != _WORLD_SIZE:
            raise ValueError(f"Cake MXFP8 MegaMoE requires EP16, got EP{world_size}")
        device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (10, 3):
            major, minor = torch.cuda.get_device_capability(device)
            raise RuntimeError(
                f"Cake MXFP8 MegaMoE requires compute capability 10.3, got {major}.{minor}"
            )
        self._validate_weights(weights, device=device)
        self.weights = weights
        self._routing = _prepare_balanced_routing(
            topk_ids,
            process_group=self._group,
            rank=self.rank,
            device=device,
        )
        self.tokens = int(topk_ids.shape[0])
        self._routing_ids_ptr = int(topk_ids.data_ptr())
        self._w13 = weights.w13.view(torch.uint8)
        self._w13_scale = weights.w13_scale.view(_LOCAL_EXPERTS, -1, 32, 32)
        self._w2 = weights.w2.view(torch.uint8)
        self._w2_scale = weights.w2_scale.view(_LOCAL_EXPERTS, -1, 16, 32)

        group_name = self._group.group_name
        _enable_symm_mem_for_group(group_name)
        symm_mem.set_backend("NVSHMEM")
        if str(symm_mem.get_backend(device)).upper() != "NVSHMEM":
            raise RuntimeError("Cake MXFP8 MegaMoE requires NVSHMEM symmetric memory")
        self._workspace = _Workspace(
            device=device,
            group_name=group_name,
            routing=self._routing,
        )
        self._output = self._workspace.output.tensor[: self.tokens]
        self._module = load_cake_mxfp8_megamoe_ep16_module(device=device)
        self._persistent_grid = min(
            128,
            int(torch.cuda.get_device_properties(device).multi_processor_count) & ~1,
        )

    @staticmethod
    def _validate_weights(
        weights: CakeMxfp8MegaMoeEp16Weights,
        *,
        device: torch.device,
    ) -> None:
        if not isinstance(weights, CakeMxfp8MegaMoeEp16Weights):
            raise TypeError("weights must be CakeMxfp8MegaMoeEp16Weights")
        _require_tensor(
            weights.w13,
            name="weights.w13",
            shape=(_LOCAL_EXPERTS, _FC1_ROWS, _HIDDEN),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        _require_tensor(
            weights.w13_scale,
            name="weights.w13_scale",
            shape=(_LOCAL_EXPERTS * (_FC1_ROWS // 256) * (_HIDDEN // 128) * 1024,),
            dtype=torch.uint8,
            device=device,
        )
        _require_tensor(
            weights.w2,
            name="weights.w2",
            shape=(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        _require_tensor(
            weights.w2_scale,
            name="weights.w2_scale",
            shape=(_LOCAL_EXPERTS * (_HIDDEN // 128) * (_INTERMEDIATE // 128) * 512,),
            dtype=torch.uint8,
            device=device,
        )

    @property
    def workspace_output(self) -> torch.Tensor:
        """Return the fixed-token caller-visible symmetric output view."""

        return self._output

    def run(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Submit one allocation-free forward on the current CUDA stream."""

        tokens = int(hidden_states.shape[0])
        if tokens != self.tokens:
            raise ValueError(
                f"session was prepared for {self.tokens} tokens, got {tokens}"
            )
        device = self.weights.w13.device
        _require_tensor(
            hidden_states,
            name="hidden_states",
            shape=(tokens, _HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        _require_tensor(
            topk_ids,
            name="topk_ids",
            shape=(tokens, _TOP_K),
            dtype=torch.int64,
            device=device,
        )
        if int(topk_ids.data_ptr()) != self._routing_ids_ptr:
            raise ValueError(
                "topk_ids must be the immutable tensor prepared by this session"
            )
        _require_tensor(
            topk_weights,
            name="topk_weights",
            shape=(tokens, _TOP_K),
            dtype=torch.float32,
            device=device,
        )
        _require_tensor(
            out,
            name="out",
            shape=(tokens, _HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        if int(out.data_ptr()) != int(self.workspace_output.data_ptr()):
            raise ValueError("out must alias session.workspace_output")

        workspace = self._workspace
        self._module.run(
            hidden_states,
            topk_ids,
            topk_weights,
            workspace.hidden.tensor,
            workspace.hidden_scale.tensor,
            workspace.topk_ids.tensor,
            workspace.topk_weights.tensor,
            tokens,
            _WORLD_SIZE,
            self.rank,
            workspace.flags.peers,
            workspace.input_ready.tensor,
            workspace.input_ready.peers,
            tokens,
            1,
            1,
            self._w13,
            workspace.fc1_input.view(torch.uint8),
            self._w13_scale,
            workspace.fc1_input_scale.view(_ROUTE_GROUPS, -1, 16, 32),
            workspace.fc2_input,
            workspace.fc2_input_scale,
            workspace.fc1_input.view(torch.uint8),
            workspace.fc1_input_scale,
            workspace.route_map,
            workspace.route_weights,
            workspace.work_source_ranks,
            workspace.work_source_tokens,
            workspace.work_local_groups,
            workspace.work_local_rows,
            workspace.fc1_tile_expert,
            workspace.fc1_tile_m,
            workspace.expert_row_offsets,
            workspace.ones,
            workspace.infinities,
            workspace.ones,
            workspace.zeros,
            self._w2,
            workspace.fc2_input.view(torch.uint8),
            self._w2_scale,
            workspace.fc2_input_scale.view(_LOCAL_EXPERTS, -1, 16, 32),
            workspace.local_terms,
            workspace.fc2_tile_expert,
            workspace.fc2_tile_m,
            workspace.fc1_done,
            0,
            tokens * _TOP_K,
            workspace.work_count,
            tokens,
            _FC1_ROWS,
            _HIDDEN // 128,
            _LOCAL_EXPERTS * (_INTERMEDIATE // 128),
            1,
            _ROWS_PER_EXPERT,
            _HIDDEN,
            _INTERMEDIATE // 128,
            _LOCAL_EXPERTS * (_HIDDEN // 128),
            _WORLD_SIZE,
            self.rank,
            workspace.flags.peers,
            workspace.hidden.tensor,
            workspace.hidden.peers,
            workspace.hidden_scale.tensor,
            workspace.hidden_scale.peers,
            workspace.topk_weights.tensor,
            workspace.topk_weights.peers,
            workspace.input_ready.tensor,
            workspace.input_ready.peers,
            _WORLD_SIZE,
            self.rank,
            workspace.flags.peers,
            workspace.token_back_ready.tensor,
            workspace.token_back_ready.peers,
            _WORLD_SIZE,
            self.rank,
            workspace.flags.peers,
            workspace.output.tensor,
            workspace.output.peers,
            workspace.tma_workspace,
            self._persistent_grid,
            1,
            1,
            workspace.token_back_ready.tensor,
            out,
            1,
            1,
            1,
        )
        return out


__all__ = [
    "CakeMxfp8MegaMoeEp16",
    "CakeMxfp8MegaMoeEp16Weights",
    "preprocess_cake_mxfp8_megamoe_ep16_weights",
]
