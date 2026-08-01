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

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Optional

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit import gen_sm120_direct_fused_moe_module
from flashinfer.trace.templates.moe import sm120_direct_fused_moe_trace
from flashinfer.utils import (
    _get_cache_buf,
    backend_requirement,
    register_custom_op,
    supported_compute_capability,
)


_TUNED_LAUNCHES = {
    # Qwen3.5-35B-A3B: hidden=2048, expert intermediate=512.
    (2048, 512): {
        1: (1, 448),
        2: (1, 256),
        3: (1, 576),
        4: (1, 704),
        5: (2, 896),
        6: (2, 576),
        7: (4, 704),
        8: (4, 768),
    },
    # JoyAI-LLM-Flash: hidden=2048, expert intermediate=768.
    (2048, 768): {
        1: (1, 576),
        2: (1, 896),
        3: (1, 576),
        4: (1, 768),
        5: (2, 896),
        6: (2, 576),
        7: (2, 384),
        8: (2, 704),
    },
}


def sm120_direct_fused_moe_workspace(
    num_tokens: int,
    topk: int,
    intermediate_size: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Allocate the BF16 intermediate buffer required by the direct kernel."""
    if not 1 <= num_tokens <= 8:
        raise ValueError(f"num_tokens must be in [1, 8], got {num_tokens}")
    if not 1 <= topk <= 8:
        raise ValueError(f"topk must be in [1, 8], got {topk}")
    if intermediate_size < 8 or intermediate_size > 1024 or intermediate_size % 8:
        raise ValueError(
            "intermediate_size must be a multiple of 8 in [8, 1024], "
            f"got {intermediate_size}"
        )
    return torch.empty(
        (num_tokens * topk, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )


def _recommended_launch(
    num_tokens: int, hidden_size: int, intermediate_size: int
) -> tuple[int, int]:
    policy = _TUNED_LAUNCHES.get((hidden_size, intermediate_size))
    if policy is not None:
        return policy[num_tokens]
    return 1, 256


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    ndim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape={tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(
            f"{name} must be on the same device as hidden_states "
            f"({tensor.device} vs {device})"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@supported_compute_capability([120])
def _check_sm120_direct_fused_moe_supported(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    expert_map: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    outputs_per_warp: Optional[int] = None,
    num_threads: Optional[int] = None,
) -> bool:
    if hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must be 2D [num_tokens, hidden_size], "
            f"got shape={tuple(hidden_states.shape)}"
        )
    if not hidden_states.is_cuda:
        raise ValueError("hidden_states must be a CUDA tensor")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(
            f"hidden_states must have dtype torch.bfloat16, got {hidden_states.dtype}"
        )
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")

    num_tokens, hidden_size = hidden_states.shape
    if not 1 <= num_tokens <= 8:
        raise ValueError(f"num_tokens must be in [1, 8], got {num_tokens}")
    if hidden_size < 8 or hidden_size > 8192 or hidden_size % 8:
        raise ValueError(
            f"hidden_size must be a multiple of 8 in [8, 8192], got {hidden_size}"
        )
    device = hidden_states.device

    _check_tensor("topk_ids", topk_ids, ndim=2, dtype=torch.int32, device=device)
    _check_tensor(
        "topk_weights", topk_weights, ndim=2, dtype=torch.float32, device=device
    )
    _check_tensor(
        "gemm1_weights", gemm1_weights, ndim=3, dtype=torch.bfloat16, device=device
    )
    _check_tensor(
        "gemm2_weights", gemm2_weights, ndim=3, dtype=torch.bfloat16, device=device
    )

    if topk_ids.shape[0] != num_tokens:
        raise ValueError("topk_ids.shape[0] must equal num_tokens")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights must have the same shape as topk_ids")
    topk = topk_ids.shape[1]
    if not 1 <= topk <= 8:
        raise ValueError(f"topk must be in [1, 8], got {topk}")

    num_local_experts = gemm1_weights.shape[0]
    if num_local_experts < 1:
        raise ValueError("at least one local expert is required")
    intermediate_size = gemm2_weights.shape[2]
    if intermediate_size < 8 or intermediate_size > 1024 or intermediate_size % 8:
        raise ValueError(
            "intermediate_size must be a multiple of 8 in [8, 1024], "
            f"got {intermediate_size}"
        )
    if gemm1_weights.shape != (
        num_local_experts,
        2 * intermediate_size,
        hidden_size,
    ):
        raise ValueError(
            "gemm1_weights must have shape "
            f"({num_local_experts}, {2 * intermediate_size}, {hidden_size})"
        )
    if gemm2_weights.shape != (
        num_local_experts,
        hidden_size,
        intermediate_size,
    ):
        raise ValueError(
            "gemm2_weights must have shape "
            f"({num_local_experts}, {hidden_size}, {intermediate_size})"
        )

    if expert_map is not None:
        _check_tensor(
            "expert_map", expert_map, ndim=1, dtype=torch.int32, device=device
        )
        if expert_map.numel() and expert_map.numel() < num_local_experts:
            raise ValueError("expert_map must be empty or a global-to-local map")
    if output is not None:
        _check_tensor("output", output, ndim=2, dtype=torch.bfloat16, device=device)
        if output.shape != hidden_states.shape:
            raise ValueError("output must have the same shape as hidden_states")
    if workspace is not None:
        _check_tensor(
            "workspace", workspace, ndim=2, dtype=torch.bfloat16, device=device
        )
        expected = (num_tokens * topk, intermediate_size)
        if workspace.shape != expected:
            raise ValueError(
                f"workspace must have shape {expected}, got {tuple(workspace.shape)}"
            )

    default_outputs, default_threads = _recommended_launch(
        num_tokens, hidden_size, intermediate_size
    )
    launch_outputs = default_outputs if outputs_per_warp is None else outputs_per_warp
    launch_threads = default_threads if num_threads is None else num_threads
    if launch_outputs not in (1, 2, 4, 8):
        raise ValueError("outputs_per_warp must be one of 1, 2, 4, or 8")
    if launch_threads < 64 or launch_threads > 1024 or launch_threads % 32:
        raise ValueError("num_threads must be a warp multiple in [64, 1024]")
    return True


@functools.cache
def get_sm120_direct_fused_moe_module():
    """Build, load, and cache the SM120 direct fused MoE JIT module."""
    module = gen_sm120_direct_fused_moe_module().build_and_load()

    @register_custom_op(
        "flashinfer::sm120_direct_fused_moe",
        mutates_args=["intermediate", "output"],
    )
    def run(
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        expert_map: torch.Tensor,
        intermediate: torch.Tensor,
        output: torch.Tensor,
        outputs_per_warp: int,
        num_threads: int,
    ) -> None:
        module.sm120_direct_fused_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm2_weights,
            expert_map,
            intermediate,
            output,
            outputs_per_warp,
            num_threads,
        )

    return SimpleNamespace(run=run)


@backend_requirement({}, common_check=_check_sm120_direct_fused_moe_supported)
@flashinfer_api(trace=sm120_direct_fused_moe_trace)
def sm120_direct_fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    expert_map: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    outputs_per_warp: Optional[int] = None,
    num_threads: Optional[int] = None,
) -> torch.Tensor:
    r"""Run a low-token BF16 SwiGLU MoE directly from precomputed routes.

    The kernel is specialized for SM120 decode batches with ``1 <= M <= 8``.
    It avoids token sorting and gathering: one kernel computes the routed
    gate/up projection and SwiGLU, and a second kernel computes the down
    projection while accumulating top-k routes in FP32.

    ``gemm1_weights`` uses FlashInfer's unquantized layout ``[up || gate]``.
    ``expert_map`` optionally maps global expert ids to rank-local ids; entries
    with value ``-1`` are skipped so the result is the rank-local EP partial.

    Pass reusable ``output`` and ``workspace`` tensors to make execution
    allocation-free and CUDA Graph safe. If omitted, the output is allocated
    and a process-local cached workspace is used.

    Parameters
    ----------
    hidden_states : torch.Tensor
        BF16 tensor with shape ``[num_tokens, hidden_size]``.
    topk_ids : torch.Tensor
        Precomputed expert ids with shape ``[num_tokens, topk]`` and int32 dtype.
    topk_weights : torch.Tensor
        Routing weights with shape ``[num_tokens, topk]`` and float32 dtype.
    gemm1_weights : torch.Tensor
        Local BF16 weights with shape ``[num_local_experts, 2 * I, H]`` in
        ``[up || gate]`` order.
    gemm2_weights : torch.Tensor
        Local BF16 weights with shape ``[num_local_experts, H, I]``.
    expert_map : torch.Tensor, optional
        Int32 global-to-local expert map. Negative entries denote remote experts.
    output : torch.Tensor, optional
        Reusable BF16 output buffer with shape ``[num_tokens, hidden_size]``.
    workspace : torch.Tensor, optional
        Reusable BF16 buffer with shape ``[num_tokens * topk, I]``.
    outputs_per_warp, num_threads : int, optional
        Launch overrides. Defaults use measured SM120 policies for
        ``H=2048, I in {512, 768}`` and a conservative fallback otherwise.

    Returns
    -------
    torch.Tensor
        BF16 rank-local MoE output with shape ``[num_tokens, hidden_size]``.
    """
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    intermediate_size = gemm2_weights.shape[2]
    if output is None:
        output = torch.empty_like(hidden_states)
    if workspace is None:
        workspace_items = num_tokens * topk * intermediate_size
        raw_workspace = _get_cache_buf(
            "sm120_direct_fused_moe_workspace",
            workspace_items * torch.bfloat16.itemsize,
            hidden_states.device,
        )
        workspace = raw_workspace[: workspace_items * torch.bfloat16.itemsize].view(
            torch.bfloat16
        )
        workspace = workspace.reshape(num_tokens * topk, intermediate_size)
    if expert_map is None:
        raw_map = _get_cache_buf(
            "sm120_direct_fused_moe_empty_expert_map", 4, hidden_states.device
        )
        expert_map = raw_map[:0].view(torch.int32)

    default_outputs, default_threads = _recommended_launch(
        num_tokens, hidden_size, intermediate_size
    )
    launch_outputs = default_outputs if outputs_per_warp is None else outputs_per_warp
    launch_threads = default_threads if num_threads is None else num_threads
    get_sm120_direct_fused_moe_module().run(
        hidden_states,
        topk_ids,
        topk_weights,
        gemm1_weights,
        gemm2_weights,
        expert_map,
        workspace,
        output,
        launch_outputs,
        launch_threads,
    )
    return output
