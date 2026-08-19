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

Low-latency block-FP8 MoE specialized for the GLM5 decode shape.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.moe import glm5_low_latency_moe_trace
from ..utils import backend_requirement, supported_compute_capability


_NUM_EXPERTS = 256
_NUM_EXPERTS_WITH_SHARED = 257
_TOP_K = 8
_HIDDEN_SIZE = 6144
_MAX_TOKENS = 4
_CTA_OUT_ROWS = 64
_NUM_K_ITER = 8
_UP_TILE_BYTES = 49152
_UP_COMBINED_TILE_BYTES = 2 * _UP_TILE_BYTES
_UP_M_TILES_PER_CTA = 4
_UP_COMBINED_M_TILES_PER_CTA = 8
_UP_ROW_HALVES_PER_M_TILE = 2
_UP_ROWS_PER_HALF = 8
_UP_K_SIXTHS_PER_ITER = 6
_UP_K_SUBS_PER_SIXTH = 4
_UP_COL_HALVES_PER_K_SUB = 2
_UP_COL_QUADS_PER_HALF = 4
_UP_BYTES_PER_COL_QUAD = 4


@dataclass(frozen=True)
class Glm5LowLatencyMoeWeights:
    """Prepared tensors accepted by :func:`glm5_low_latency_moe`."""

    expert_gate_up_weight: torch.Tensor
    expert_gate_up_scale: torch.Tensor
    routed_down_weight: torch.Tensor
    routed_down_scale: torch.Tensor
    shared_down_weight: torch.Tensor
    shared_down_scale: torch.Tensor

    def as_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "expert_gate_up_weight": self.expert_gate_up_weight,
            "expert_gate_up_scale": self.expert_gate_up_scale,
            "routed_down_weight": self.routed_down_weight,
            "routed_down_scale": self.routed_down_scale,
            "shared_down_weight": self.shared_down_weight,
            "shared_down_scale": self.shared_down_scale,
        }


@dataclass(frozen=True)
class Glm5LowLatencyMoeWorkspace:
    """Reusable output buffers for allocation-free decode calls."""

    topk_weights: torch.Tensor
    topk_indices: torch.Tensor
    expert_slots: torch.Tensor


def alloc_glm5_low_latency_moe_workspace(
    num_tokens: int,
    local_intermediate_size: int,
    device: torch.device | str,
) -> Glm5LowLatencyMoeWorkspace:
    """Allocate the temporary buffers used by :func:`glm5_low_latency_moe`."""
    if not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(
            f"GLM5 low-latency MoE supports 1 <= M <= {_MAX_TOKENS}, got {num_tokens}."
        )
    if local_intermediate_size not in (256, 512):
        raise ValueError("local_intermediate_size must be 256 (TP8) or 512 (TP4).")
    return Glm5LowLatencyMoeWorkspace(
        topk_weights=torch.empty(
            (num_tokens, _TOP_K), dtype=torch.float32, device=device
        ),
        topk_indices=torch.empty(
            (num_tokens, _TOP_K), dtype=torch.int32, device=device
        ),
        expert_slots=torch.empty(
            (num_tokens, _TOP_K + 1, local_intermediate_size),
            dtype=torch.float16,
            device=device,
        ),
    )


def _pack_up_weight_side(weight: torch.Tensor) -> torch.Tensor:
    """Pack ``[..., I, H]`` FP8 rows into the expert-up MMA lane layout."""
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "GLM5 low-latency MoE expects float8_e4m3fn gate/up weights, "
            f"got {weight.dtype}."
        )
    if weight.shape[-1] != _HIDDEN_SIZE:
        raise ValueError(
            f"GLM5 low-latency MoE expects hidden size {_HIDDEN_SIZE}, "
            f"got {weight.shape[-1]}."
        )
    inter_per_tp = weight.shape[-2]
    if inter_per_tp not in (256, 512):
        raise ValueError(
            "GLM5 low-latency MoE expects local intermediate size 256 (TP8) or "
            f"512 (TP4), got {inter_per_tp}."
        )

    prefix_shape = tuple(weight.shape[:-2])
    prefix_dims = len(prefix_shape)
    sub_rows = inter_per_tp // _CTA_OUT_ROWS
    reshaped = weight.contiguous().reshape(
        *prefix_shape,
        sub_rows,
        _UP_M_TILES_PER_CTA,
        _UP_ROW_HALVES_PER_M_TILE,
        _UP_ROWS_PER_HALF,
        _NUM_K_ITER,
        _UP_K_SIXTHS_PER_ITER,
        _UP_K_SUBS_PER_SIXTH,
        _UP_COL_HALVES_PER_K_SUB,
        _UP_COL_QUADS_PER_HALF,
        _UP_BYTES_PER_COL_QUAD,
    )
    order = (
        *range(prefix_dims),
        prefix_dims,
        prefix_dims + 4,
        prefix_dims + 5,
        prefix_dims + 1,
        prefix_dims + 6,
        prefix_dims + 3,
        prefix_dims + 8,
        prefix_dims + 7,
        prefix_dims + 2,
        prefix_dims + 9,
    )
    return (
        reshaped.permute(order)
        .contiguous()
        .reshape(*prefix_shape, sub_rows, _NUM_K_ITER, _UP_TILE_BYTES)
    )


def _interleave_packed_gate_up(
    gate_packed: torch.Tensor, up_packed: torch.Tensor
) -> torch.Tensor:
    if gate_packed.shape != up_packed.shape:
        raise ValueError(
            "Packed gate/up shapes must match, got "
            f"{gate_packed.shape} and {up_packed.shape}."
        )
    prefix_shape = tuple(gate_packed.shape[:-3])
    sub_rows = gate_packed.shape[-3]
    side_shape = (
        *prefix_shape,
        sub_rows,
        _NUM_K_ITER,
        _UP_K_SIXTHS_PER_ITER,
        _UP_M_TILES_PER_CTA,
        _UP_K_SUBS_PER_SIXTH,
        _UP_ROWS_PER_HALF,
        _UP_COL_QUADS_PER_HALF,
        _UP_COL_HALVES_PER_K_SUB,
        _UP_ROW_HALVES_PER_M_TILE,
        _UP_BYTES_PER_COL_QUAD,
    )
    combined = torch.empty(
        (*prefix_shape, sub_rows, _NUM_K_ITER, _UP_COMBINED_TILE_BYTES),
        device=gate_packed.device,
        dtype=gate_packed.dtype,
    )
    combined_view = combined.reshape(
        *prefix_shape,
        sub_rows,
        _NUM_K_ITER,
        _UP_K_SIXTHS_PER_ITER,
        _UP_COMBINED_M_TILES_PER_CTA,
        _UP_K_SUBS_PER_SIXTH,
        _UP_ROWS_PER_HALF,
        _UP_COL_QUADS_PER_HALF,
        _UP_COL_HALVES_PER_K_SUB,
        _UP_ROW_HALVES_PER_M_TILE,
        _UP_BYTES_PER_COL_QUAD,
    )
    gate_view = gate_packed.reshape(side_shape)
    up_view = up_packed.reshape(side_shape)
    for worker_m in range(_UP_COMBINED_M_TILES_PER_CTA):
        old_m = worker_m // _UP_ROW_HALVES_PER_M_TILE
        old_row_half = worker_m % _UP_ROW_HALVES_PER_M_TILE
        combined_view[..., :, worker_m, :, :, :, :, 0, :].copy_(
            gate_view[..., :, old_m, :, :, :, :, old_row_half, :]
        )
        combined_view[..., :, worker_m, :, :, :, :, 1, :].copy_(
            up_view[..., :, old_m, :, :, :, :, old_row_half, :]
        )
    return combined


def pack_glm5_low_latency_moe_gate_up_weight(
    shared_gate_up_weight: torch.Tensor,
    routed_up_gate_weight: torch.Tensor,
) -> torch.Tensor:
    """Pack shared ``[gate, up]`` and routed ``[up, gate]`` FP8 weights.

    The result has shape ``[257, I/64, 8, 98304]``. Row zero is the shared
    expert and rows 1..256 are routed experts.
    """
    if shared_gate_up_weight.device != routed_up_gate_weight.device:
        raise ValueError(
            "Shared and routed gate/up weights must be on the same device."
        )
    if shared_gate_up_weight.ndim != 2 or shared_gate_up_weight.shape[0] % 2:
        raise ValueError(
            "shared_gate_up_weight must have shape [2 * I, 6144], got "
            f"{tuple(shared_gate_up_weight.shape)}."
        )
    if (
        routed_up_gate_weight.ndim != 3
        or routed_up_gate_weight.shape[0] != _NUM_EXPERTS
    ):
        raise ValueError(
            "routed_up_gate_weight must have shape [256, 2 * I, 6144], got "
            f"{tuple(routed_up_gate_weight.shape)}."
        )
    shared_gate, shared_up = shared_gate_up_weight.chunk(2, dim=0)
    routed_up, routed_gate = routed_up_gate_weight.chunk(2, dim=1)
    if shared_gate.shape[-2:] != routed_gate.shape[-2:]:
        raise ValueError(
            "Shared and routed local intermediate sizes must match, got "
            f"{shared_gate.shape[0]} and {routed_gate.shape[1]}."
        )

    shared_combined = _interleave_packed_gate_up(
        _pack_up_weight_side(shared_gate), _pack_up_weight_side(shared_up)
    )
    routed_combined = _interleave_packed_gate_up(
        _pack_up_weight_side(routed_gate), _pack_up_weight_side(routed_up)
    )
    packed = torch.empty(
        (_NUM_EXPERTS_WITH_SHARED, *shared_combined.shape),
        device=shared_gate_up_weight.device,
        dtype=shared_gate_up_weight.dtype,
    )
    packed[0].copy_(shared_combined)
    packed[1:].copy_(routed_combined)
    return packed


def pack_glm5_low_latency_moe_gate_up_scale(
    shared_gate_up_scale: torch.Tensor,
    routed_up_gate_scale: torch.Tensor,
) -> torch.Tensor:
    """Combine shared ``[gate, up]`` and routed ``[up, gate]`` FP32 scales."""
    if shared_gate_up_scale.device != routed_up_gate_scale.device:
        raise ValueError("Shared and routed gate/up scales must be on the same device.")
    if shared_gate_up_scale.ndim != 2 or shared_gate_up_scale.shape[0] % 2:
        raise ValueError("shared_gate_up_scale must have shape [2 * I/128, 48].")
    if (
        routed_up_gate_scale.ndim != 3
        or routed_up_gate_scale.shape[0] != _NUM_EXPERTS
        or tuple(routed_up_gate_scale.shape[1:]) != tuple(shared_gate_up_scale.shape)
    ):
        raise ValueError("routed_up_gate_scale must have shape [256, 2 * I/128, 48].")
    if (
        shared_gate_up_scale.dtype != torch.float32
        or routed_up_gate_scale.dtype != torch.float32
    ):
        raise TypeError("GLM5 low-latency MoE gate/up scales must be float32.")

    scale_rows = shared_gate_up_scale.shape[0] // 2
    packed = torch.empty(
        (_NUM_EXPERTS_WITH_SHARED, *shared_gate_up_scale.shape),
        device=shared_gate_up_scale.device,
        dtype=torch.float32,
    )
    packed[0].copy_(shared_gate_up_scale)
    packed[1:, :scale_rows].copy_(routed_up_gate_scale[:, scale_rows:])
    packed[1:, scale_rows:].copy_(routed_up_gate_scale[:, :scale_rows])
    return packed


def prepare_glm5_low_latency_moe_weights(
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    routed_up_gate_weight: torch.Tensor,
    routed_up_gate_scale: torch.Tensor,
    routed_down_weight: torch.Tensor,
    routed_down_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
) -> Glm5LowLatencyMoeWeights:
    """Prepare raw GLM5 TP4/TP8 block-FP8 weights for repeated decode calls."""
    return Glm5LowLatencyMoeWeights(
        expert_gate_up_weight=pack_glm5_low_latency_moe_gate_up_weight(
            shared_gate_up_weight, routed_up_gate_weight
        ),
        expert_gate_up_scale=pack_glm5_low_latency_moe_gate_up_scale(
            shared_gate_up_scale, routed_up_gate_scale
        ),
        routed_down_weight=routed_down_weight.contiguous(),
        routed_down_scale=routed_down_scale.contiguous(),
        shared_down_weight=shared_down_weight.contiguous(),
        shared_down_scale=shared_down_scale.contiguous(),
    )


@functools.cache
def _get_glm5_low_latency_moe_module():
    from ..jit.glm5_moe import load_glm5_low_latency_moe_module

    return load_glm5_low_latency_moe_module()


def _check_glm5_low_latency_moe_shapes(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    expert_gate_up_weight: torch.Tensor,
    expert_gate_up_scale: torch.Tensor,
    routed_down_weight: torch.Tensor,
    routed_down_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
) -> tuple[int, int]:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [M, 6144].")
    num_tokens = hidden_states.shape[0]
    if not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(
            f"GLM5 low-latency MoE supports 1 <= M <= {_MAX_TOKENS}, got {num_tokens}."
        )
    if tuple(hidden_states.shape) != (num_tokens, _HIDDEN_SIZE):
        raise ValueError("hidden_states must have shape [M, 6144].")
    if tuple(router_logits.shape) != (num_tokens, _NUM_EXPERTS):
        raise ValueError("router_logits must have shape [M, 256].")
    if tuple(routing_bias.shape) != (_NUM_EXPERTS,):
        raise ValueError("routing_bias must have shape [256].")
    if (
        expert_gate_up_weight.ndim != 4
        or expert_gate_up_weight.shape[0] != _NUM_EXPERTS_WITH_SHARED
        or expert_gate_up_weight.shape[2] != _NUM_K_ITER
        or expert_gate_up_weight.shape[3] != _UP_COMBINED_TILE_BYTES
    ):
        raise ValueError("expert_gate_up_weight must have shape [257, I/64, 8, 98304].")
    inter_per_tp = expert_gate_up_weight.shape[1] * _CTA_OUT_ROWS
    if inter_per_tp not in (256, 512):
        raise ValueError("local intermediate size must be 256 (TP8) or 512 (TP4).")
    if tuple(expert_gate_up_scale.shape) != (
        _NUM_EXPERTS_WITH_SHARED,
        2 * (inter_per_tp // 128),
        _HIDDEN_SIZE // 128,
    ):
        raise ValueError("expert_gate_up_scale has an incompatible shape.")
    if tuple(routed_down_weight.shape) != (
        _NUM_EXPERTS,
        _HIDDEN_SIZE,
        inter_per_tp,
    ):
        raise ValueError("routed_down_weight must have shape [256, 6144, I].")
    if tuple(routed_down_scale.shape) != (
        _NUM_EXPERTS,
        _HIDDEN_SIZE // 128,
        inter_per_tp // 128,
    ):
        raise ValueError("routed_down_scale must have shape [256, 48, I/128].")
    if tuple(shared_down_weight.shape) != (_HIDDEN_SIZE, inter_per_tp):
        raise ValueError("shared_down_weight must have shape [6144, I].")
    if tuple(shared_down_scale.shape) != (
        _HIDDEN_SIZE // 128,
        inter_per_tp // 128,
    ):
        raise ValueError("shared_down_scale must have shape [48, I/128].")
    return num_tokens, inter_per_tp


@supported_compute_capability([100, 103])
def _check_glm5_low_latency_moe_supported(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    expert_gate_up_weight: torch.Tensor,
    expert_gate_up_scale: torch.Tensor,
    routed_down_weight: torch.Tensor,
    routed_down_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    routed_scaling_factor: float = 2.5,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[Glm5LowLatencyMoeWorkspace] = None,
) -> bool:
    num_tokens, inter_per_tp = _check_glm5_low_latency_moe_shapes(
        hidden_states,
        router_logits,
        routing_bias,
        expert_gate_up_weight,
        expert_gate_up_scale,
        routed_down_weight,
        routed_down_scale,
        shared_down_weight,
        shared_down_scale,
    )
    tensors = (
        hidden_states,
        router_logits,
        routing_bias,
        expert_gate_up_weight,
        expert_gate_up_scale,
        routed_down_weight,
        routed_down_scale,
        shared_down_weight,
        shared_down_scale,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("All GLM5 low-latency MoE inputs must be CUDA tensors.")
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError(
            "All GLM5 low-latency MoE inputs must be on the same CUDA device."
        )
    if hidden_states.dtype != torch.bfloat16 or routing_bias.dtype != torch.bfloat16:
        raise TypeError("hidden_states and routing_bias must be bfloat16.")
    if router_logits.dtype != torch.float32:
        raise TypeError("router_logits must be float32.")
    if expert_gate_up_weight.dtype != torch.float8_e4m3fn:
        raise TypeError("expert_gate_up_weight must be float8_e4m3fn.")
    if (
        routed_down_weight.dtype != torch.float8_e4m3fn
        or shared_down_weight.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("Down-projection weights must be float8_e4m3fn.")
    if any(
        tensor.dtype != torch.float32
        for tensor in (
            expert_gate_up_scale,
            routed_down_scale,
            shared_down_scale,
        )
    ):
        raise TypeError("GLM5 low-latency MoE scales must be float32.")
    if out is not None and (
        out.device != hidden_states.device
        or out.dtype != torch.bfloat16
        or tuple(out.shape) != (num_tokens, _HIDDEN_SIZE)
    ):
        raise ValueError("out must be bfloat16 [M, 6144] on the input device.")
    if workspace is not None:
        expected = (
            (workspace.topk_weights, (num_tokens, _TOP_K), torch.float32),
            (workspace.topk_indices, (num_tokens, _TOP_K), torch.int32),
            (
                workspace.expert_slots,
                (num_tokens, _TOP_K + 1, inter_per_tp),
                torch.float16,
            ),
        )
        if any(
            tensor.device != hidden_states.device
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            for tensor, shape, dtype in expected
        ):
            raise ValueError(
                "workspace tensors have incompatible shapes, dtypes, or devices."
            )
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
        raise ValueError("routed_scaling_factor must be positive and finite.")
    return True


@backend_requirement({}, common_check=_check_glm5_low_latency_moe_supported)
@flashinfer_api(trace=glm5_low_latency_moe_trace)
def glm5_low_latency_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    expert_gate_up_weight: torch.Tensor,
    expert_gate_up_scale: torch.Tensor,
    routed_down_weight: torch.Tensor,
    routed_down_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    routed_scaling_factor: float = 2.5,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[Glm5LowLatencyMoeWorkspace] = None,
) -> torch.Tensor:
    """Run the GLM5 low-latency block-FP8 MoE path on SM100/SM103.

    This specialized decode kernel supports 256 routed experts, one shared
    expert, top-8 sigmoid routing, hidden size 6144, ``M <= 4``, and local
    intermediate size 256 (TP8) or 512 (TP4). Call
    :func:`prepare_glm5_low_latency_moe_weights` once when loading model weights.

    The returned tensor is this TP rank's local contribution. Distributed
    callers must all-reduce it across TP ranks before the residual connection.

    Parameters
    ----------
    hidden_states : torch.Tensor
        BF16 input with shape ``[M, 6144]``, where ``1 <= M <= 4``.
    router_logits : torch.Tensor
        FP32 routed-expert logits with shape ``[M, 256]``.
    routing_bias : torch.Tensor
        BF16 no-aux routing bias with shape ``[256]``. The bias affects expert
        selection; normalized expert weights use the unbiased sigmoid scores.
    expert_gate_up_weight : torch.Tensor
        Packed FP8 gate/up weights from
        :func:`pack_glm5_low_latency_moe_gate_up_weight`.
    expert_gate_up_scale : torch.Tensor
        Packed FP32 block scales from
        :func:`pack_glm5_low_latency_moe_gate_up_scale`.
    routed_down_weight, shared_down_weight : torch.Tensor
        Raw row-major FP8 down-projection weights.
    routed_down_scale, shared_down_scale : torch.Tensor
        FP32 128x128 down-projection block scales.
    routed_scaling_factor : float
        Scale applied after normalizing the selected sigmoid scores.
    out : Optional[torch.Tensor]
        Optional BF16 output buffer with shape ``[M, 6144]``.
    workspace : Optional[Glm5LowLatencyMoeWorkspace]
        Reusable temporaries allocated by
        :func:`alloc_glm5_low_latency_moe_workspace`. Supplying this and ``out``
        makes repeated decode calls allocation-free.
    Returns
    -------
    torch.Tensor
        BF16 TP-local MoE contribution with shape ``[M, 6144]``.
    """
    num_tokens, inter_per_tp = _check_glm5_low_latency_moe_shapes(
        hidden_states,
        router_logits,
        routing_bias,
        expert_gate_up_weight,
        expert_gate_up_scale,
        routed_down_weight,
        routed_down_scale,
        shared_down_weight,
        shared_down_scale,
    )
    if out is None:
        out = torch.empty(
            (num_tokens, _HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
    if workspace is None:
        workspace = alloc_glm5_low_latency_moe_workspace(
            num_tokens, inter_per_tp, hidden_states.device
        )

    module = _get_glm5_low_latency_moe_module()
    module.glm5_fused_expert_up(
        router_logits,
        hidden_states,
        routing_bias,
        expert_gate_up_weight,
        expert_gate_up_scale,
        workspace.topk_weights,
        workspace.topk_indices,
        workspace.expert_slots,
        float(routed_scaling_factor),
    )
    module.glm5_fused_expert_down(
        workspace.expert_slots,
        workspace.topk_indices,
        workspace.topk_weights,
        routed_down_weight,
        routed_down_scale,
        shared_down_weight,
        shared_down_scale,
        out,
    )
    return out
