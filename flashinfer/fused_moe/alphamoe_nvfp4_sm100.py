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

import functools
import math

import torch

from ..api_logging import flashinfer_api
from ..jit import gen_alphamoe_nvfp4_sm100_module
from ..trace.templates.moe import alphamoe_nvfp4_aligned_moe_trace
from ..utils import (
    backend_requirement,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

_SUPPORTED_CC = [100, 103]
_ROUTE_SUBTILE = 8
_UP_BLOCK_K = 256
_W1_ROWS = 256
_INT32_MAX = 2**31 - 1


def _require_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ndim: int,
    contiguous: bool = True,
) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@supported_compute_capability(_SUPPORTED_CC)
def _check_alphamoe_nvfp4_supported(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int = 8,
    routed_scaling_factor: float = 1.0,
) -> bool:
    """Validate the frozen schedule's complete host-visible contract."""

    _require_cuda_tensor(
        "hidden_states",
        hidden_states,
        dtype=torch.uint8,
        ndim=2,
        contiguous=False,
    )
    _require_cuda_tensor(
        "hidden_states_scale",
        hidden_states_scale,
        dtype=torch.float8_e4m3fn,
        ndim=2,
    )
    _require_cuda_tensor("gemm1_weights", gemm1_weights, dtype=torch.uint8, ndim=3)
    _require_cuda_tensor(
        "gemm1_weights_scale",
        gemm1_weights_scale,
        dtype=torch.float8_e4m3fn,
        ndim=3,
    )
    _require_cuda_tensor("gemm2_weights", gemm2_weights, dtype=torch.uint8, ndim=3)
    _require_cuda_tensor(
        "gemm2_weights_scale",
        gemm2_weights_scale,
        dtype=torch.float8_e4m3fn,
        ndim=3,
    )
    _require_cuda_tensor(
        "sorted_token_ids", sorted_token_ids, dtype=torch.int32, ndim=1
    )
    _require_cuda_tensor("expert_ids", expert_ids, dtype=torch.int32, ndim=1)
    _require_cuda_tensor(
        "num_tokens_post_padded",
        num_tokens_post_padded,
        dtype=torch.int32,
        ndim=1,
    )
    _require_cuda_tensor("topk_weights", topk_weights, dtype=torch.float32, ndim=2)
    _require_cuda_tensor("out", out, dtype=torch.bfloat16, ndim=2)

    device = hidden_states.device
    for name, tensor in (
        ("hidden_states_scale", hidden_states_scale),
        ("gemm1_weights", gemm1_weights),
        ("gemm1_weights_scale", gemm1_weights_scale),
        ("gemm2_weights", gemm2_weights),
        ("gemm2_weights_scale", gemm2_weights_scale),
        ("sorted_token_ids", sorted_token_ids),
        ("expert_ids", expert_ids),
        ("num_tokens_post_padded", num_tokens_post_padded),
        ("topk_weights", topk_weights),
        ("out", out),
    ):
        if tensor.device != device:
            raise ValueError(
                f"{name} must be on the same device as hidden_states "
                f"({tensor.device} vs {device})"
            )

    if (
        hidden_states.stride(-1) != 1
        or hidden_states.stride(0) <= 0
        or hidden_states.stride(0) < hidden_states.shape[1]
        or hidden_states.stride(0) % 16 != 0
    ):
        raise ValueError(
            "hidden_states must have unit innermost stride and a positive, "
            "non-overlapping, 16-byte-aligned row stride"
        )
    if hidden_states.data_ptr() % 16 != 0:
        raise ValueError("hidden_states data pointer must be 16-byte aligned for TMA")
    for name, tensor in (
        ("gemm1_weights", gemm1_weights),
        ("gemm2_weights", gemm2_weights),
    ):
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} data pointer must be 16-byte aligned for TMA")

    m, packed_k = hidden_states.shape
    k = 2 * packed_k
    num_experts, n, w1_packed_k = gemm1_weights.shape
    if m <= 0:
        raise ValueError("hidden_states must contain at least one token")
    if k < _UP_BLOCK_K or k % _UP_BLOCK_K != 0:
        raise ValueError(
            f"logical hidden size K ({k}) must be at least {_UP_BLOCK_K} "
            f"and divisible by {_UP_BLOCK_K}"
        )
    if n < _W1_ROWS or n % _W1_ROWS != 0:
        raise ValueError(
            f"gemm1_weights.shape[1] ({n}) must be at least {_W1_ROWS} "
            f"and divisible by {_W1_ROWS}"
        )
    if num_experts <= 0:
        raise ValueError("gemm1_weights must contain at least one expert")
    if w1_packed_k != packed_k:
        raise ValueError(
            "gemm1_weights.shape[2] must equal hidden_states.shape[1] "
            f"({w1_packed_k} vs {packed_k})"
        )

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must not exceed num_experts ({num_experts})")
    if (
        not isinstance(block_m, int)
        or isinstance(block_m, bool)
        or block_m < _ROUTE_SUBTILE
        or block_m % _ROUTE_SUBTILE != 0
    ):
        raise ValueError(
            f"block_m must be a positive multiple of {_ROUTE_SUBTILE}, got {block_m!r}"
        )
    if not math.isfinite(routed_scaling_factor):
        raise ValueError(
            f"routed_scaling_factor must be finite, got {routed_scaling_factor}"
        )

    intermediate = n // 2
    expected_shapes = {
        "hidden_states_scale": (m, k // 16),
        "gemm1_weights_scale": (num_experts, n, k // 16),
        "gemm2_weights": (num_experts, k, intermediate // 2),
        "gemm2_weights_scale": (num_experts, k, intermediate // 16),
        "topk_weights": (m, top_k),
        "out": (m, k),
    }
    actual_tensors = {
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "topk_weights": topk_weights,
        "out": out,
    }
    for name, expected in expected_shapes.items():
        actual = tuple(actual_tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}")
    if out.data_ptr() % 16 != 0:
        raise ValueError("out data pointer must be 16-byte aligned for bulk reduction")

    if num_tokens_post_padded.numel() != 1:
        raise ValueError(
            "num_tokens_post_padded must contain exactly one device-side int32 value"
        )
    if expert_ids.numel() <= 0:
        raise ValueError("expert_ids must not be empty")
    required_plan_capacity = expert_ids.numel() * block_m
    if sorted_token_ids.numel() < required_plan_capacity:
        raise ValueError(
            "sorted_token_ids capacity must be at least "
            f"expert_ids.numel() * block_m ({required_plan_capacity}), got "
            f"{sorted_token_ids.numel()}"
        )
    int_index_extents = {
        "M": m,
        "K": k,
        "M * top_k": m * top_k,
        "M * K": m * k,
        "hidden_states_scale.numel()": hidden_states_scale.numel(),
        "gemm1_weights_scale.numel()": gemm1_weights_scale.numel(),
        "gemm2_weights_scale.numel()": gemm2_weights_scale.numel(),
        "routing plan capacity": required_plan_capacity,
    }
    for name, extent in int_index_extents.items():
        if extent > _INT32_MAX:
            raise ValueError(f"{name} ({extent}) must fit in signed int32")
    grid_x = expert_ids.numel() * (block_m // _ROUTE_SUBTILE)
    if grid_x > _INT32_MAX:
        raise ValueError(f"launch grid.x ({grid_x}) exceeds the CUDA limit")
    if n // _W1_ROWS > 65535:
        raise ValueError(f"launch grid.y ({n // _W1_ROWS}) exceeds the CUDA limit")
    return True


@functools.cache
def get_alphamoe_nvfp4_sm100_module():
    """Build and cache the frozen AlphaMoE NVFP4 TVM-FFI module."""

    return gen_alphamoe_nvfp4_sm100_module().build_and_load()


@register_custom_op("flashinfer::alphamoe_nvfp4_aligned_moe", mutates_args=("out",))
def _alphamoe_nvfp4_aligned_moe_impl(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int,
    routed_scaling_factor: float,
) -> None:
    get_alphamoe_nvfp4_sm100_module().nvfp4_aligned_moe_op(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        routed_scaling_factor,
    )


@register_fake_op("flashinfer::alphamoe_nvfp4_aligned_moe")
def _alphamoe_nvfp4_aligned_moe_fake(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int,
    routed_scaling_factor: float,
) -> None:
    pass


@backend_requirement({}, common_check=_check_alphamoe_nvfp4_supported)
@flashinfer_api(trace=alphamoe_nvfp4_aligned_moe_trace)
def alphamoe_nvfp4_aligned_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int = 8,
    routed_scaling_factor: float = 1.0,
) -> None:
    r"""Run the fused AlphaMoE NVFP4 up → SwiGLU → down kernel.

    This SM100/SM103 kernel consumes a pre-aligned routing plan and accumulates
    directly into the caller-owned BF16 ``out`` tensor. It does not run expert
    selection and does not allocate an intermediate or output tensor.

    ``hidden_states``, ``gemm1_weights``, and ``gemm2_weights`` store two E2M1
    values per ``uint8`` byte, with the even logical value in the low nibble.
    Their E4M3 scales are linear, contiguous, and cover 16 logical values each:

    - ``hidden_states``: ``[M, K / 2]``; scale ``[M, K / 16]``
    - ``gemm1_weights``: ``[E, N, K / 2]`` in conventional ``[gate; up]`` row
      order; scale ``[E, N, K / 16]``
    - ``gemm2_weights``: ``[E, K, N / 4]``; scale ``[E, K, N / 32]``

    The scale tensors must use ``torch.float8_e4m3fn`` and the linear per-16
    layout above. FlashInfer's 128x4-swizzled NVFP4 scale layout is a different
    contract and must not be passed to this kernel.

    ``sorted_token_ids`` and ``expert_ids`` follow the aligned MoE plan used by
    vLLM/SGLang. ``num_tokens_post_padded`` is a one-element device tensor
    naming the valid plan extent; blocks in the capacity-sized launch grid that
    lie past this extent are skipped. The caller must keep that device value no
    larger than ``expert_ids.numel() * block_m`` and provide valid expert and
    token ids in the active plan.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Packed E2M1 activations ``[M, K / 2]``. The innermost stride must be 1;
        row-strided views are supported.
    hidden_states_scale : torch.Tensor
        Linear E4M3 scales ``[M, K / 16]``.
    gemm1_weights : torch.Tensor
        Packed gate/up weights ``[E, N, K / 2]`` with ``N`` divisible by 256.
    gemm1_weights_scale : torch.Tensor
        Linear E4M3 scales ``[E, N, K / 16]``.
    gemm2_weights : torch.Tensor
        Packed down weights ``[E, K, N / 4]``.
    gemm2_weights_scale : torch.Tensor
        Linear E4M3 scales ``[E, K, N / 32]``.
    sorted_token_ids : torch.Tensor
        Contiguous int32 aligned-plan entries.
    expert_ids : torch.Tensor
        Contiguous int32 expert id per ``block_m`` plan entries.
    num_tokens_post_padded : torch.Tensor
        One-element device int32 valid plan extent.
    topk_weights : torch.Tensor
        FP32 route weights ``[M, top_k]``.
    out : torch.Tensor
        Contiguous BF16 accumulator ``[M, K]``. Contributions are added to its
        existing values; its data pointer must be 16-byte aligned. Zero it
        before calling when a fresh result is wanted. It must not overlap any
        input tensor.
    top_k : int
        Routes per token.
    block_m : int
        Routing plan block size, at least 8 and divisible by 8.
    routed_scaling_factor : float
        Finite scalar applied to each routed contribution.

    Notes
    -----
    Logical ``K`` is derived as ``2 * hidden_states.shape[1]`` and must be at
    least 256 and divisible by 256. This function mutates ``out`` and returns
    ``None``.
    """

    _alphamoe_nvfp4_aligned_moe_impl(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        routed_scaling_factor,
    )


__all__ = ["alphamoe_nvfp4_aligned_moe", "get_alphamoe_nvfp4_sm100_module"]
