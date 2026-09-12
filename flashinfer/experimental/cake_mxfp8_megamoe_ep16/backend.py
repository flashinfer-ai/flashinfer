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
import tvm_ffi

from ...comm.torch_symmetric_memory import _enable_symm_mem_for_group
from ...moe_ep.cake_mxfp8_megamoe_ep16 import CakeMxfp8MegaMoeEp16Weights
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
_EXPERT_ROW_SEGMENTS = (32, 16, 16)
_EXPERT_ROWS = sum(_EXPERT_ROW_SEGMENTS)
_SUPPORTED_TOKENS = (16, 32, _MAX_TOKENS_PER_RANK)
_FUSED_GRID_CTAS = 144
_MAX_LAUNCH_EPOCH = (2**31 - 1) // _FUSED_GRID_CTAS - 1


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


def _interleave_gate_up_16(w13: torch.Tensor) -> torch.Tensor:
    experts, rows, hidden = w13.shape
    if rows % 32:
        raise ValueError("gate/up rows must be divisible by 32")
    intermediate = rows // 2
    result = torch.empty_like(w13)
    result_blocks = result.view(
        experts,
        intermediate // 16,
        2,
        16,
        hidden,
    )
    gate = w13[:, :intermediate].view(experts, intermediate // 16, 16, hidden)
    up = w13[:, intermediate:].view(experts, intermediate // 16, 16, hidden)
    result_blocks[:, :, 0].copy_(gate)
    result_blocks[:, :, 1].copy_(up)
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
        safe_max = torch.clamp(block_max, min=1.0e-30)
        scale_exp = torch.ceil(torch.log2(safe_max * (1.0 / 448.0)))
        codes = torch.clamp(scale_exp + 127.0, min=0.0, max=254.0).to(torch.int32)
        codes = torch.where(block_max == 0, torch.zeros_like(codes), codes)
        decoded = torch.pow(2.0, codes.float() - 127.0)
        block_quantized = (values / decoded.unsqueeze(-1)).to(torch.float8_e4m3fn)
        quantized[:, row_begin:row_end].copy_(
            block_quantized.reshape(experts, row_end - row_begin, columns)
        )
        scale_codes[:, row_begin:row_end].copy_(codes.to(torch.uint8))
    return quantized, scale_codes


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
    interleaved_w13 = _interleave_gate_up_16(w13)
    w13_fp8, w13_scale_plain = _quantize_mxfp8_block32(interleaved_w13)
    w2_fp8, w2_scale_plain = _quantize_mxfp8_block32(w2)
    return CakeMxfp8MegaMoeEp16Weights(
        w13=w13_fp8,
        w13_scale=_pack_scale_n128_k128(w13_scale_plain),
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


def _validate_gathered_routing_capacity(gathered: torch.Tensor) -> None:
    if not bool(torch.all((gathered >= 0) & (gathered < _EXPERTS)).item()):
        raise ValueError(f"topk_ids must contain expert IDs in [0, {_EXPERTS})")
    route_counts = torch.bincount(gathered.reshape(-1), minlength=_EXPERTS)
    max_expert_load = int(route_counts.max().item())
    if max_expert_load > _EXPERT_ROWS:
        raise ValueError(
            "EP16 routing exceeds the per-expert capacity: "
            f"maximum load is {max_expert_load}, capacity is {_EXPERT_ROWS}"
        )


def _validate_launch_epoch(epoch: int) -> None:
    if epoch < 0 or epoch > _MAX_LAUNCH_EPOCH:
        raise RuntimeError(
            "Cake MXFP8 MegaMoE session launch epoch is exhausted; "
            "create a new session before submitting another forward"
        )


def _validate_routing_capacity(
    topk_ids: torch.Tensor,
    *,
    process_group: dist.ProcessGroup,
    device: torch.device,
) -> None:
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
    _validate_gathered_routing_capacity(gathered)


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
        tokens: int,
    ) -> None:
        self.flags = _allocate_symmetric(
            (2,),
            torch.uint32,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.flags.tensor.zero_()
        self.published_hidden = _allocate_symmetric(
            (tokens, _HIDDEN),
            torch.bfloat16,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.published_topk_ids = _allocate_symmetric(
            (tokens, _TOP_K),
            torch.int32,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.published_topk_weights = _allocate_symmetric(
            (tokens, _TOP_K),
            torch.float32,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )
        self.route_terms = _allocate_symmetric(
            (tokens, _TOP_K, _HIDDEN),
            torch.bfloat16,
            device=device,
            group_name=group_name,
            world_size=_WORLD_SIZE,
        )

        self.activation_bf16 = torch.empty(
            (_LOCAL_EXPERTS, _EXPERT_ROWS, _HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        self.fc1_workspace_bf16 = torch.empty(
            (_LOCAL_EXPERTS, _EXPERT_ROWS, _INTERMEDIATE),
            dtype=torch.bfloat16,
            device=device,
        )
        self.fc2_output_bf16 = torch.empty(
            (_LOCAL_EXPERTS, _EXPERT_ROWS, _HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        self.route_map_i32 = torch.empty(
            (_LOCAL_EXPERTS, _EXPERT_ROWS),
            dtype=torch.int32,
            device=device,
        )
        self.route_scale_f32 = torch.empty(
            (_LOCAL_EXPERTS, _EXPERT_ROWS),
            dtype=torch.float32,
            device=device,
        )
        self.route_counts_u32 = torch.zeros(
            (_LOCAL_EXPERTS,),
            dtype=torch.uint32,
            device=device,
        )
        self.fc1_done = torch.zeros(
            (_LOCAL_EXPERTS * len(_EXPERT_ROW_SEGMENTS),),
            dtype=torch.uint32,
            device=device,
        )
        self.publication_done = torch.zeros(1, dtype=torch.uint32, device=device)
        self.publication_visible = torch.zeros(1, dtype=torch.uint32, device=device)
        self.dispatch_done = torch.zeros(1, dtype=torch.uint32, device=device)
        self.compute_done = torch.zeros(1, dtype=torch.uint32, device=device)
        self.return_done = torch.zeros(1, dtype=torch.uint32, device=device)
        self.return_visible = torch.zeros(1, dtype=torch.uint32, device=device)
        self.output_bf16 = torch.empty(
            (tokens, _HIDDEN), dtype=torch.bfloat16, device=device
        )
        self.tma_backing, self.tma_workspace = _aligned_workspace(1024, device=device)

    def destroy(self) -> None:
        """Release resources when the owning session is discarded."""


class CakeMxfp8MegaMoeEp16:
    """Prepared EP16 MXFP8 MegaMoE session for exact SM103a devices.

    The route supports 512 experts, hidden size 3072, intermediate size 5120,
    top-k 8, 16/32/64 tokens per rank, and up to 64 routes per expert. Routing
    is validated once at construction and must remain immutable. Construction
    owns all symmetric memory and scratch storage. A session must be used
    serially from one CUDA stream. :meth:`run` submits without allocating, and
    CUDA Graph capture is not supported.
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
        _validate_routing_capacity(
            topk_ids,
            process_group=self._group,
            device=device,
        )
        self.tokens = int(topk_ids.shape[0])
        self._routing_ids_ptr = int(topk_ids.data_ptr())
        self._w13 = weights.w13.view(torch.uint8)
        self._w13_scale = weights.w13_scale.view(torch.uint8).reshape(-1, 128)
        self._w2 = weights.w2.view(torch.uint8)
        self._w2_scale = weights.w2_scale.view(torch.uint8).reshape(-1, 128)

        group_name = self._group.group_name
        _enable_symm_mem_for_group(group_name)
        symm_mem.set_backend("NVSHMEM")
        if str(symm_mem.get_backend(device)).upper() != "NVSHMEM":
            raise RuntimeError("Cake MXFP8 MegaMoE requires NVSHMEM symmetric memory")
        self._workspace = _Workspace(
            device=device,
            group_name=group_name,
            tokens=self.tokens,
        )
        self._output = self._workspace.output_bf16
        self._module = load_cake_mxfp8_megamoe_ep16_module(device=device)
        with torch.cuda.device(device), tvm_ffi.use_torch_stream():
            self._module.setup_tma(
                self._w13,
                self._w13_scale,
                self._workspace.activation_bf16,
                self._w2,
                self._w2_scale,
                self._workspace.fc1_workspace_bf16,
                self._workspace.tma_workspace,
            )
        self._launch_epoch = 0
        torch.cuda.synchronize(device=device)
        dist.barrier(group=self._group)

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
            shape=(_LOCAL_EXPERTS * _FC1_ROWS * (_HIDDEN // 32),),
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
            shape=(_LOCAL_EXPERTS * _HIDDEN * (_INTERMEDIATE // 32),),
            dtype=torch.uint8,
            device=device,
        )

    @property
    def workspace_output(self) -> torch.Tensor:
        """Return the fixed-token caller-visible output tensor."""

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

        device = self.weights.w13.device
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Cake MXFP8 MegaMoE does not support CUDA Graph capture"
                )
        tokens = int(hidden_states.shape[0])
        if tokens != self.tokens:
            raise ValueError(
                f"session was prepared for {self.tokens} tokens, got {tokens}"
            )
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
        launch_epoch = self._launch_epoch
        _validate_launch_epoch(launch_epoch)
        with torch.cuda.device(device), tvm_ffi.use_torch_stream():
            self._module.run(
                hidden_states,
                topk_ids,
                topk_weights,
                self._w13,
                self._w13_scale,
                workspace.activation_bf16,
                self._w2,
                self._w2_scale,
                workspace.fc1_workspace_bf16,
                workspace.fc2_output_bf16,
                workspace.route_map_i32,
                workspace.route_scale_f32,
                workspace.route_counts_u32,
                workspace.fc1_done,
                workspace.publication_done,
                workspace.publication_visible,
                workspace.dispatch_done,
                workspace.compute_done,
                workspace.return_done,
                workspace.return_visible,
                launch_epoch,
                tokens,
                _WORLD_SIZE,
                self.rank,
                workspace.flags.peers,
                workspace.published_hidden.tensor,
                workspace.published_hidden.peers,
                workspace.published_topk_ids.tensor,
                workspace.published_topk_ids.peers,
                workspace.published_topk_weights.tensor,
                workspace.published_topk_weights.peers,
                workspace.route_terms.tensor,
                workspace.route_terms.peers,
                out,
                workspace.tma_workspace,
            )
        self._launch_epoch = launch_epoch + 1
        return out


__all__ = [
    "CakeMxfp8MegaMoeEp16",
    "CakeMxfp8MegaMoeEp16Weights",
    "preprocess_cake_mxfp8_megamoe_ep16_weights",
]
