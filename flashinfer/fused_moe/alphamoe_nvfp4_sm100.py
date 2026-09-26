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
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..jit import gen_alphamoe_nvfp4_sm100_module
from ..trace.templates.moe import (
    alphamoe_nvfp4_aligned_moe_trace,
    alphamoe_nvfp4_routed_moe_trace,
)
from ..utils import (
    backend_requirement,
    supported_compute_capability,
)

_SUPPORTED_CC = [100, 103]
_ROUTE_SUBTILE = 8
_UP_BLOCK_K = 256
# Largest token count the route-9 decode chain admits beyond M=8. The in-kernel alignment of its up
# kernel holds M * top_k <= 256 pairs (M <= 32 for top-8); the admitted band is set from measurement.
_ROUTE9_MAX_M = 16
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
    output1_scale_gate_scalar: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int = 8,
    routed_scaling_factor: float = 1.0,
    w1_scale_prepared: Optional[torch.Tensor] = None,
    w2_scale_prepared: Optional[torch.Tensor] = None,
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
        "output1_scale_gate_scalar",
        output1_scale_gate_scalar,
        dtype=torch.float32,
        ndim=1,
    )
    _require_cuda_tensor(
        "output1_scale_scalar",
        output1_scale_scalar,
        dtype=torch.float32,
        ndim=1,
    )
    _require_cuda_tensor(
        "output2_scale_scalar",
        output2_scale_scalar,
        dtype=torch.float32,
        ndim=1,
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
        ("output1_scale_gate_scalar", output1_scale_gate_scalar),
        ("output1_scale_scalar", output1_scale_scalar),
        ("output2_scale_scalar", output2_scale_scalar),
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
        "output1_scale_gate_scalar": (num_experts,),
        "output1_scale_scalar": (num_experts,),
        "output2_scale_scalar": (num_experts,),
        "topk_weights": (m, top_k),
        "out": (m, k),
    }
    actual_tensors = {
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "output1_scale_gate_scalar": output1_scale_gate_scalar,
        "output1_scale_scalar": output1_scale_scalar,
        "output2_scale_scalar": output2_scale_scalar,
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
    """Build and cache the AlphaMoE NVFP4 TVM-FFI module."""

    return gen_alphamoe_nvfp4_sm100_module().build_and_load()


@torch.library.custom_op(
    "flashinfer::alphamoe_nvfp4_aligned_moe", mutates_args=("out",)
)
def _alphamoe_nvfp4_aligned_moe_impl(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int,
    routed_scaling_factor: float,
    w1_scale_prepared: Optional[torch.Tensor] = None,
    w2_scale_prepared: Optional[torch.Tensor] = None,
) -> None:
    # Seed from the caller's output to preserve additive semantics. This
    # temporary and both casts use the current stream; during graph capture
    # its storage belongs to PyTorch's graph memory pool.
    accumulator = out.to(torch.float32)
    get_alphamoe_nvfp4_sm100_module().nvfp4_current_aligned_moe_op(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_gate_scalar,
        output1_scale_scalar,
        output2_scale_scalar,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        accumulator,
        top_k,
        block_m,
        routed_scaling_factor,
        w1_scale_prepared,
        w2_scale_prepared,
    )
    out.copy_(accumulator)


@torch.library.register_fake("flashinfer::alphamoe_nvfp4_aligned_moe")
def _alphamoe_nvfp4_aligned_moe_fake(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int,
    routed_scaling_factor: float,
    w1_scale_prepared: Optional[torch.Tensor] = None,
    w2_scale_prepared: Optional[torch.Tensor] = None,
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
    output1_scale_gate_scalar: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int = 8,
    routed_scaling_factor: float = 1.0,
    w1_scale_prepared: Optional[torch.Tensor] = None,
    w2_scale_prepared: Optional[torch.Tensor] = None,
) -> None:
    r"""Run AlphaMoE NVFP4 gate/up, SwiGLU, requantization and down compute.

    This SM100/SM103 operator consumes a pre-aligned routing plan. Contributions
    accumulate in a temporary FP32 ``[M, K]`` buffer initialized from the
    caller-owned BF16 ``out`` tensor, then convert back to ``out`` once. It does
    not run expert selection. The temporary uses ``4 * M * K`` bytes and is
    compatible with CUDA graph capture.

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
    output1_scale_gate_scalar : torch.Tensor
        Contiguous FP32 per-expert gate dequantization scales ``[E]``. Each
        gate accumulator is multiplied by its routed expert's value before
        applying SiLU.
    output1_scale_scalar : torch.Tensor
        Contiguous FP32 per-expert up-projection scales ``[E]``. For static
        ModelOpt FP4 this is the up-projection global scale divided by the
        second-activation quantization scale.
    output2_scale_scalar : torch.Tensor
        Contiguous FP32 per-expert down-projection scales ``[E]``. Each down
        accumulator is multiplied by its routed expert's value before route
        weighting and ``routed_scaling_factor``.
    sorted_token_ids : torch.Tensor
        Contiguous int32 aligned-plan entries.
    expert_ids : torch.Tensor
        Contiguous int32 expert id per ``block_m`` plan entries.
    num_tokens_post_padded : torch.Tensor
        One-element device int32 valid plan extent.
    topk_weights : torch.Tensor
        FP32 route weights ``[M, top_k]``.
    out : torch.Tensor
        Contiguous BF16 output ``[M, K]``. Contributions are added to its existing
        values in FP32 before the final BF16 conversion; its data pointer must
        be 16-byte aligned. Zero it before calling when a fresh result is wanted.
        It must not overlap any input tensor.
    top_k : int
        Routes per token.
    block_m : int
        Routing plan block size, at least 8 and divisible by 8.
    routed_scaling_factor : float
        Finite scalar applied to each routed contribution.

    w1_scale_prepared, w2_scale_prepared : Optional[torch.Tensor]
        Immutable uint8 panels prepared once from the raw scale tensors
        with prepare_nvfp4_w1_scales/prepare_nvfp4_w2_scales before
        requests or graph capture. Raw tensors remain required.

    Notes
    -----
    Logical ``K`` is derived as ``2 * hidden_states.shape[1]`` and must be at
    least 256 and divisible by 256. This function mutates ``out`` and returns
    ``None``. The FP32 bulk reduction is order-dependent and flushes subnormal
    inputs and results to signed zero, as specified by PTX
    ``cp.reduce.async.bulk.add.f32``.
    """

    _alphamoe_nvfp4_aligned_moe_impl(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_gate_scalar,
        output1_scale_scalar,
        output2_scale_scalar,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        routed_scaling_factor,
        w1_scale_prepared,
        w2_scale_prepared,
    )


__all__ = [
    "alphamoe_nvfp4_aligned_moe",
    "alphamoe_nvfp4_routed_moe",
    "get_alphamoe_nvfp4_sm100_module",
]


# Small raw-ID alignment and output-seed companion; aligned API above is unchanged.
def is_alphamoe_nvfp4_small_alignment_seed_supported(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    initial_out: torch.Tensor,
    pad_sorted_token_ids: bool = True,
) -> bool:
    """Metadata-only dispatch predicate; callers retain their general fallback."""
    return (
        topk_ids.is_cuda
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and topk_ids.ndim == 2
        and topk_ids.shape[0] in (1, 8)
        and topk_ids.shape[1] == 8
        and num_experts in (257, 513)
        and block_size in (8, 16)
        and pad_sorted_token_ids
        and initial_out.is_cuda
        and initial_out.dtype == torch.bfloat16
        and initial_out.is_contiguous()
        and initial_out.ndim == 2
        and initial_out.shape[0] == topk_ids.shape[0]
        and initial_out.shape[1] > 0
        and initial_out.device == topk_ids.device
    )


@torch.library.custom_op(
    "flashinfer::alphamoe_nvfp4_align_and_seed_output",
    mutates_args=(
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_padded",
        "cumsum_buffer",
        "seeded_accumulator",
    ),
)
def _alphamoe_nvfp4_align_and_seed_output_impl(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    initial_out: torch.Tensor,
    seeded_accumulator: torch.Tensor,
    pad_sorted_token_ids: bool,
) -> None:
    get_alphamoe_nvfp4_sm100_module().nvfp4_align_and_seed_output_op(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        initial_out,
        seeded_accumulator,
        pad_sorted_token_ids,
    )


@torch.library.register_fake("flashinfer::alphamoe_nvfp4_align_and_seed_output")
def _alphamoe_nvfp4_align_and_seed_output_fake(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    initial_out: torch.Tensor,
    seeded_accumulator: torch.Tensor,
    pad_sorted_token_ids: bool,
) -> None:
    pass


def alphamoe_nvfp4_align_and_seed_output(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    initial_out: torch.Tensor,
    seeded_accumulator: torch.Tensor,
    pad_sorted_token_ids: bool = True,
) -> None:
    """Fill a small route plan and convert the caller BF16 seed to private FP32.

    This SM100/SM103 companion accepts contiguous int32 CUDA IDs of shape
    (1, 8) or (8, 8), a reserved-inclusive bucket count of 257 or 513, and
    block size 8 or 16. IDs follow the existing convention: expert -1 occupies
    reserved bucket zero, and other IDs are in [0, num_experts - 2].

    All route-plan outputs are caller-owned contiguous int32 tensors on the same device.
    sorted_token_ids must have capacity at least topk_ids.numel()*block_size;
    expert_ids needs at least topk_ids.numel() entries. The padded extent has
    one entry and cumsum_buffer at least num_experts+1 entries. The full sorted
    capacity is padded with the numel sentinel. Unused expert entries remain
    unchanged. cumsum_buffer receives each bucket start plus its actual count,
    followed by the final padded extent. Intra-expert ordering is unspecified.

    initial_out is read-only BF16 [M,K] and seeded_accumulator is a disjoint
    contiguous FP32 tensor of the identical shape/device. Every seed element
    is converted, including nonzero values. Pass that same accumulator to
    the subsequent aligned compute; do not repeat its output conversion.

    Select the existing general alignment operator for other metadata.
    The existing aligned compute entry and its caller-provided plan remain
    independently usable.
    """
    _require_cuda_tensor("topk_ids", topk_ids, dtype=torch.int32, ndim=2)
    for name, tensor in (
        ("sorted_token_ids", sorted_token_ids),
        ("expert_ids", expert_ids),
        ("num_tokens_post_padded", num_tokens_post_padded),
        ("cumsum_buffer", cumsum_buffer),
    ):
        _require_cuda_tensor(name, tensor, dtype=torch.int32, ndim=1)
        if tensor.device != topk_ids.device:
            raise ValueError(f"{name} must be on the same device as topk_ids")
    if topk_ids.shape[0] not in (1, 8) or topk_ids.shape[1] != 8:
        raise ValueError("topk_ids must have shape (1, 8) or (8, 8)")
    if num_experts not in (257, 513) or block_size not in (8, 16):
        raise ValueError(
            "alignment requires 257/513 reserved-inclusive bins and block size 8/16"
        )
    if not pad_sorted_token_ids:
        raise ValueError("alignment requires full-capacity sorted-token padding")
    pairs = topk_ids.numel()
    if (
        sorted_token_ids.numel() < pairs * block_size
        or sorted_token_ids.numel() > _INT32_MAX
    ):
        raise ValueError(
            "sorted_token_ids capacity must cover every routed pair's padded block and fit int32"
        )
    if expert_ids.numel() < pairs:
        raise ValueError("expert_ids must have at least topk_ids.numel() entries")
    if num_tokens_post_padded.numel() != 1:
        raise ValueError("num_tokens_post_padded must have one entry")
    if cumsum_buffer.numel() < num_experts + 1:
        raise ValueError("cumsum_buffer must have at least num_experts + 1 entries")
    _require_cuda_tensor("initial_out", initial_out, dtype=torch.bfloat16, ndim=2)
    _require_cuda_tensor(
        "seeded_accumulator", seeded_accumulator, dtype=torch.float32, ndim=2
    )
    if not is_alphamoe_nvfp4_small_alignment_seed_supported(
        topk_ids, num_experts, block_size, initial_out, pad_sorted_token_ids
    ):
        raise ValueError(
            "alignment and seed require supported route metadata and matching BF16 output"
        )
    if (
        seeded_accumulator.shape != initial_out.shape
        or seeded_accumulator.device != initial_out.device
    ):
        raise ValueError("seeded_accumulator must match the output shape and device")
    _alphamoe_nvfp4_align_and_seed_output_impl(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        initial_out,
        seeded_accumulator,
        pad_sorted_token_ids,
    )


@torch.library.custom_op(
    "flashinfer::alphamoe_nvfp4_routed_moe",
    mutates_args=(
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_padded",
        "cumsum_buffer",
        "out",
    ),
)
def _alphamoe_nvfp4_routed_moe_impl(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    topk_ids: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    top_k: int,
    block_m: int,
    routed_scaling_factor: float,
    w1_scale_prepared: Optional[torch.Tensor] = None,
    w2_scale_prepared: Optional[torch.Tensor] = None,
    w1_data_prepared: Optional[torch.Tensor] = None,
    w1_gate_up_data_prepared: Optional[torch.Tensor] = None,
    w1_gate_up_scale_prepared: Optional[torch.Tensor] = None,
    w1_scale_prepared_interleaved: Optional[torch.Tensor] = None,
    w2_data_prepared: Optional[torch.Tensor] = None,
    accumulate: bool = True,
) -> None:
    if _alphamoe_try_complete_routed(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_gate_scalar,
        output1_scale_scalar,
        output2_scale_scalar,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        topk_ids,
        cumsum_buffer,
        top_k,
        block_m,
        routed_scaling_factor,
        w1_scale_prepared,
        w1_data_prepared,
        w1_gate_up_data_prepared,
        w1_gate_up_scale_prepared,
        w2_scale_prepared,
        w1_scale_prepared_interleaved,
        w2_data_prepared,
        accumulate,
    ):
        return
    if not accumulate:
        # The remaining (non-complete) paths seed from the caller output.
        out.zero_()
    # The private buffer is seeded by the alignment kernel on this stream.
    # During graph capture it belongs to PyTorch's graph memory pool.
    accumulator = torch.empty_like(out, dtype=torch.float32)
    module = get_alphamoe_nvfp4_sm100_module()
    routed_op = (
        module.nvfp4_current_general_routed_seeded_moe_op
        if _uses_general_alignment_seed_portfolio(
            hidden_states, gemm1_weights, top_k, block_m
        )
        else module.nvfp4_current_routed_seeded_moe_op
    )
    private_owners: tuple[torch.Tensor, ...] = ()
    if _uses_compact_owner_routed(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights_scale,
        topk_ids,
        out,
        top_k,
        block_m,
        w1_scale_prepared,
        w2_scale_prepared,
    ):
        owner_capacity = (topk_ids.numel() + 31) // 32 + gemm1_weights.shape[0]
        owner_plan = torch.empty(
            (owner_capacity, 3), dtype=torch.int32, device=out.device
        )
        owner_count = torch.empty((1,), dtype=torch.int32, device=out.device)
        private_owners = (owner_plan, owner_count)
        routed_op = module.nvfp4_compact_owner_routed_seeded_moe_op
    routed_op(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_gate_scalar,
        output1_scale_scalar,
        output2_scale_scalar,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        accumulator,
        topk_ids,
        cumsum_buffer,
        top_k,
        block_m,
        routed_scaling_factor,
        w1_scale_prepared,
        w2_scale_prepared,
        *private_owners,
    )
    out.copy_(accumulator)


@torch.library.register_fake("flashinfer::alphamoe_nvfp4_routed_moe")
def _alphamoe_nvfp4_routed_moe_fake(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    topk_ids: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    top_k: int,
    block_m: int,
    routed_scaling_factor: float,
    w1_scale_prepared: Optional[torch.Tensor] = None,
    w2_scale_prepared: Optional[torch.Tensor] = None,
    w1_data_prepared: Optional[torch.Tensor] = None,
    w1_gate_up_data_prepared: Optional[torch.Tensor] = None,
    w1_gate_up_scale_prepared: Optional[torch.Tensor] = None,
    w1_scale_prepared_interleaved: Optional[torch.Tensor] = None,
    w2_data_prepared: Optional[torch.Tensor] = None,
    accumulate: bool = True,
) -> None:
    pass


@flashinfer_api(trace=alphamoe_nvfp4_routed_moe_trace)
def alphamoe_nvfp4_routed_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    topk_ids: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    top_k: int,
    block_m: int = 8,
    routed_scaling_factor: float = 1.0,
    w1_scale_prepared: Optional[torch.Tensor] = None,
    w2_scale_prepared: Optional[torch.Tensor] = None,
    w1_data_prepared: Optional[torch.Tensor] = None,
    w1_gate_up_data_prepared: Optional[torch.Tensor] = None,
    w1_gate_up_scale_prepared: Optional[torch.Tensor] = None,
    w1_scale_prepared_interleaved: Optional[torch.Tensor] = None,
    w2_data_prepared: Optional[torch.Tensor] = None,
    accumulate: bool = True,
) -> torch.Tensor:
    """Run route alignment, complete expert computation and weighted accumulation.

    Selected complete routes use private per-route accumulators and finalize
    contributions in route order, preserving the caller's existing output.
    The exact route controls seed initialization, FP32 or BF16 route storage,
    optional compact-owner metadata and immutable prepared W1 scales/data. Buffer
    initialization, expert computation and finalization all stay inside this
    call and use the current stream, including during CUDA graph capture.

    Inputs without a selected complete route retain the existing alignment,
    accumulator and compute path. The aligned API is unchanged. Prepared scale
    tensors remain optional immutable model-load inputs; this function performs
    no weight preparation, host device-count read or automatic model fetching.
    For the selected 8-, 128- and 512-token routes, pass ``w1_data_prepared`` from
    :func:`prepare_nvfp4_w1_data`; omitting it retains the existing path. The
    128- and 512-token routes also take ``w2_data_prepared`` from
    :func:`prepare_nvfp4_w2_data`; without both prepared data tensors those
    token counts retain the existing path.
    For the adjacent gate/up M512 route, pass both
    ``w1_gate_up_data_prepared`` and ``w1_gate_up_scale_prepared`` from the
    matching model-load preparation helpers. They use a distinct layout
    from the optional old prepared inputs. No packing occurs in this call.
    The caller-owned output is updated and returned. With ``accumulate=False`` the
    weighted route sum is written directly (the prior contents of ``out`` are
    ignored, so no zero fill is needed); the result equals the seeded path on a
    zero output bit for bit.
    """
    _check_alphamoe_nvfp4_supported(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_gate_scalar,
        output1_scale_scalar,
        output2_scale_scalar,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        routed_scaling_factor,
        w1_scale_prepared,
        w2_scale_prepared,
    )
    _check_prepared_w1_data(gemm1_weights, w1_data_prepared)
    _check_prepared_w2_data(gemm2_weights, w2_data_prepared)
    _check_prepared_w1_gate_up(
        gemm1_weights, w1_gate_up_data_prepared, w1_gate_up_scale_prepared
    )
    _check_prepared_w1_interleaved(
        gemm1_weights, gemm1_weights_scale, w1_scale_prepared_interleaved
    )
    if not (
        is_alphamoe_nvfp4_routed_seed_supported(
            hidden_states, gemm1_weights, topk_ids, out, top_k, block_m
        )
        or _alphamoe_complete_route_id(
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights_scale,
            topk_ids,
            out,
            top_k,
            block_m,
            w1_scale_prepared,
            w1_data_prepared,
            w1_gate_up_data_prepared,
            w1_gate_up_scale_prepared,
            w1_scale_prepared_interleaved,
            w2_scale_prepared,
        )
    ):
        raise ValueError("routed companion requires supported route metadata")
    _alphamoe_nvfp4_routed_moe_impl(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_gate_scalar,
        output1_scale_scalar,
        output2_scale_scalar,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        topk_ids,
        cumsum_buffer,
        top_k,
        block_m,
        routed_scaling_factor,
        w1_scale_prepared,
        w2_scale_prepared,
        w1_data_prepared,
        w1_gate_up_data_prepared,
        w1_gate_up_scale_prepared,
        w1_scale_prepared_interleaved,
        w2_data_prepared,
        accumulate,
    )
    return out


def is_alphamoe_nvfp4_alignment_seed_supported(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    initial_out: torch.Tensor,
    pad_sorted_token_ids: bool = True,
) -> bool:
    """Metadata-only dispatch predicate; callers retain their general fallback."""
    return (
        topk_ids.is_cuda
        and topk_ids.dtype == torch.int32
        and topk_ids.is_contiguous()
        and topk_ids.ndim == 2
        and topk_ids.shape[0] > 0
        and topk_ids.shape[1] > 0
        and topk_ids.numel() <= 2147483647
        and 1 < num_experts <= 1024
        and block_size in (8, 16)
        and pad_sorted_token_ids
        and initial_out.is_cuda
        and initial_out.dtype == torch.bfloat16
        and initial_out.is_contiguous()
        and initial_out.ndim == 2
        and initial_out.shape[0] == topk_ids.shape[0]
        and initial_out.shape[1] > 0
        and initial_out.device == topk_ids.device
    )


def _uses_general_alignment_seed_portfolio(
    hidden_states, gemm1_weights, top_k, block_m
):
    return hidden_states.shape[0] in (128, 512) and (
        gemm1_weights.shape[1],
        hidden_states.shape[1] * 2,
        gemm1_weights.shape[0],
        top_k,
        block_m,
    ) == (1024, 6144, 256, 8, 8)


def is_alphamoe_nvfp4_routed_seed_supported(
    hidden_states,
    gemm1_weights,
    topk_ids,
    out,
    top_k,
    block_m,
):
    """Return whether this exact compute route has a fused alignment/seed companion."""
    if _uses_general_alignment_seed_portfolio(
        hidden_states, gemm1_weights, top_k, block_m
    ):
        return is_alphamoe_nvfp4_alignment_seed_supported(
            topk_ids, gemm1_weights.shape[0] + 1, block_m, out, True
        )
    return is_alphamoe_nvfp4_small_alignment_seed_supported(
        topk_ids, gemm1_weights.shape[0] + 1, block_m, out, True
    )


def prepare_nvfp4_w1_data(w1):
    """Prepare immutable packed W1 panels once during model loading.

    The contiguous uint8 input is [E,N,K/2] in the original gate/up row order.
    The result is [E*(N/128)*(K/256),128,128], an exact byte permutation with
    no dequantization. Keep the raw weights and this tensor alive and immutable
    while using the routed operator. Preparation is outside routed execution.
    """
    if w1.dtype != torch.uint8:
        raise TypeError("W1 data must contain original packed uint8 bytes")
    if w1.ndim != 3 or not w1.is_contiguous():
        raise ValueError("W1 data must be contiguous [E,N,K/2]")
    experts, rows, packed_k = map(int, w1.shape)
    if min(experts, rows, packed_k) <= 0 or rows % 128 or packed_k % 128:
        raise ValueError("Prepared W1 data requires N%128=0 and K%256=0")
    return (
        w1.view(experts, rows // 128, 128, packed_k // 128, 128)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(-1, 128, 128)
    )


def _check_prepared_w1_data(w1, prepared):
    if prepared is None:
        return
    _require_cuda_tensor("w1_data_prepared", prepared, dtype=torch.uint8, ndim=3)
    e, n, packed_k = w1.shape
    expected = (e * (n // 128) * (packed_k // 128), 128, 128)
    if (
        prepared.device != w1.device
        or tuple(prepared.shape) != expected
        or prepared.data_ptr() % 16 != 0
    ):
        raise ValueError("Prepared W1 data must match the raw weights and panel layout")


def prepare_nvfp4_w2_data(w2):
    """Prepare immutable packed W2 panels once during model loading.

    The contiguous uint8 input is [E,K,N/4] (K output rows of N/2 packed FP4
    intermediate values per expert). The result is [E*(K/128)*(N/256),128,64],
    an exact byte permutation with no dequantization: panel
    ``(e*(K/128)+ob)*(N/256)+j`` holds rows ``ob*128..+128`` and bytes
    ``j*64..+64`` of expert ``e``. Keep the raw weights and this tensor alive and
    immutable while using the routed operator. Preparation is outside routed
    execution.
    """
    if w2.dtype != torch.uint8:
        raise TypeError("W2 data must contain original packed uint8 bytes")
    if w2.ndim != 3 or not w2.is_contiguous():
        raise ValueError("W2 data must be contiguous [E,K,N/4]")
    experts, k, packed_n = map(int, w2.shape)
    if min(experts, k, packed_n) <= 0 or k % 128 or packed_n % 64:
        raise ValueError("Prepared W2 data requires K%128=0 and N%256=0")
    return (
        w2.view(experts, k // 128, 128, packed_n // 64, 64)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(-1, 128, 64)
    )


def _check_prepared_w2_data(w2, prepared):
    if prepared is None:
        return
    _require_cuda_tensor("w2_data_prepared", prepared, dtype=torch.uint8, ndim=3)
    e, k, packed_n = w2.shape
    expected = (e * (k // 128) * (packed_n // 64), 128, 64)
    if (
        prepared.device != w2.device
        or tuple(prepared.shape) != expected
        or prepared.data_ptr() % 16 != 0
    ):
        raise ValueError("Prepared W2 data must match the raw weights and panel layout")


def prepare_nvfp4_w1_scales(w1_scale):
    """Return native CP-layout U8 panels as ``[E*(N/128)*(K/256),16,128]``.

    ``w1_scale`` keeps the original contiguous ``[E,N,K/16]`` E4M3 ABI and
    gate/up order. The result is a separately owned tensor on the same device.
    For row ``r=32*g+8*c+d`` and four-byte word ``k``, destination byte offset
    inside its 2048-byte panel is ``((4*c+k)*8+d)*16+4*g``.
    """
    import torch

    if w1_scale.dtype != torch.float8_e4m3fn:
        raise TypeError("W1 scales must contain original E4M3 bytes")
    if w1_scale.ndim != 3 or not w1_scale.is_contiguous():
        raise ValueError("W1 scales must be contiguous [E,N,K/16]")
    experts, rows, scale_columns = map(int, w1_scale.shape)
    if min(experts, rows, scale_columns) <= 0 or rows % 128 or scale_columns % 16:
        raise ValueError("Prepared W1 scales require N%128=0 and K%256=0")
    panels = rows // 128
    k_tiles = scale_columns // 16
    # Raw axes: E, panel, g, c, d, K256 tile, word, byte.
    raw = w1_scale.view(torch.uint8).reshape(experts, panels, 4, 4, 8, k_tiles, 4, 4)
    # Prepared axes: E, panel, K256 tile, c, word, d, g, byte.
    return (
        raw.permute(0, 1, 5, 3, 6, 4, 2, 7)
        .contiguous()
        .reshape(experts * panels * k_tiles, 16, 128)
    )


def prepare_nvfp4_w2_scales(w2_scale):
    """Return native CP-layout U8 panels ``[E*(K/128)*B,8,128]``.

    Raw scales are contiguous E4M3 ``[E,K,8*B]``, B=N/256. At row
    r=32*g+8*c+d, word q, byte b, the 1024-byte panel's destination is
    ``((2*c+q)*8+d)*16+4*g+b``. Original raw scales remain the oracle input.
    """
    import torch

    if w2_scale.dtype != torch.float8_e4m3fn:
        raise TypeError("W2 scales must contain original E4M3 bytes")
    if w2_scale.ndim != 3 or not w2_scale.is_contiguous():
        raise ValueError("W2 scales must be contiguous [E,K,8*B]")
    experts, rows, scale_columns = map(int, w2_scale.shape)
    if min(experts, rows, scale_columns) <= 0 or rows % 128 or scale_columns % 8:
        raise ValueError("Prepared W2 scales require K%128=0 and scale_columns%8=0")
    output_tiles, intermediate_blocks = rows // 128, scale_columns // 8
    # Original axes: E, output tile, g, c, d, intermediate block, word, byte.
    raw = w2_scale.view(torch.uint8).reshape(
        experts, output_tiles, 4, 4, 8, intermediate_blocks, 2, 4
    )
    # Native CP axes: E, output tile, intermediate block, c, word, d, g, byte.
    return (
        raw.permute(0, 1, 5, 3, 6, 4, 2, 7)
        .contiguous()
        .reshape(experts * output_tiles * intermediate_blocks, 8, 128)
    )


__all__ += [
    "prepare_nvfp4_w1_scales",
    "prepare_nvfp4_w2_scales",
    "prepare_nvfp4_w1_data",
    "prepare_nvfp4_w1_gate_up_data",
    "prepare_nvfp4_w1_gate_up_scales",
]


def _interleave_gate_up_rows(t):
    """[E, N, C] with rows = N/2 gate then N/2 up -> rows ordered (j, {gate, up}, r) for 64-row groups."""
    experts, rows, cols = map(int, t.shape)
    if rows % 128:
        raise ValueError("interleaved W1 preparation requires N%128=0")
    return (
        t.view(experts, 2, rows // 128, 64, cols)
        .permute(0, 2, 1, 3, 4)
        .reshape(experts, rows, cols)
        .contiguous()
    )


def prepare_nvfp4_w1_scales_interleaved(w1_scale):
    """CP-layout W1 scale panels ``[E*(N/128)*(K/256),16,128]`` in the gate/up-interleaved row order.

    Panel ``j`` of an expert holds the scales of gate rows ``64j..64j+63`` followed by up rows
    ``64j..64j+63``. Consumed by the eight-token complete route together with the existing
    :func:`prepare_nvfp4_w1_data` panels (no separate interleaved W1 data copy is needed); keep it
    alive and immutable while routing.
    """
    return prepare_nvfp4_w1_scales(_interleave_gate_up_rows(w1_scale))


def _check_prepared_w1_interleaved(w1, w1_scale, scales):
    if scales is None:
        return
    e, n, packed_k = w1.shape
    panels = e * (n // 128) * (packed_k // 128)
    _require_cuda_tensor("w1_scale_prepared_interleaved", scales, dtype=torch.uint8, ndim=3)
    if (
        scales.device != w1_scale.device
        or tuple(scales.shape) != (panels, 16, 128)
        or scales.data_ptr() % 16 != 0
    ):
        raise ValueError(
            "Interleaved prepared W1 scales must match the raw weights and panel layout"
        )


_S5_MERGE_COUNTERS = {}


def _s5_merge_counters(count, device):
    """Zeroed uint32 last-arriver counters for the one-token route.

    The kernels reset every counter they consume, so one zero-initialised buffer per
    (device, stream) is reused across calls without any per-call fill launch.
    """
    key = (device.index if device.index is not None else torch.cuda.current_device(), torch.cuda.current_stream(device).cuda_stream, int(count))
    buffer = _S5_MERGE_COUNTERS.get(key)
    if buffer is None:
        buffer = torch.zeros((int(count),), dtype=torch.uint32, device=device)
        _S5_MERGE_COUNTERS[key] = buffer
    return buffer


__all__ += [
    "prepare_nvfp4_w1_scales_interleaved",
]


def _uses_compact_owner_routed(
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights_scale,
    topk_ids,
    out,
    top_k,
    block_m,
    w1_scale_prepared,
    w2_scale_prepared,
):
    # Only immutable metadata selects this private routed specialization.
    m, packed_k = hidden_states.shape
    e, n, _ = gemm1_weights.shape
    k = packed_k * 2
    if (m, n, k, e, top_k, block_m) != (512, 1024, 6144, 256, 8, 8):
        return False
    if not is_alphamoe_nvfp4_alignment_seed_supported(
        topk_ids, e + 1, block_m, out, True
    ):
        return False
    if w1_scale_prepared is None or w2_scale_prepared is None:
        return False
    return (
        tuple(gemm1_weights_scale.shape) == (e, n, k // 16)
        and tuple(gemm2_weights_scale.shape) == (e, k, n // 32)
        and all(
            scale.data_ptr() % 4 == 0
            for scale in (hidden_states_scale, gemm1_weights_scale, gemm2_weights_scale)
        )
        and gemm2_weights_scale.data_ptr() % 8 == 0
        and w1_scale_prepared.dtype == torch.uint8
        and w1_scale_prepared.is_contiguous()
        and w1_scale_prepared.device == gemm1_weights_scale.device
        and tuple(w1_scale_prepared.shape) == (e * (n // 128) * (k // 256), 16, 128)
        and w1_scale_prepared.data_ptr() % 16 == 0
        and w2_scale_prepared.dtype == torch.uint8
        and w2_scale_prepared.is_contiguous()
        and w2_scale_prepared.device == gemm2_weights_scale.device
        and tuple(w2_scale_prepared.shape) == (e * (k // 128) * (n // 256), 8, 128)
        and w2_scale_prepared.data_ptr() % 16 == 0
    )


@functools.lru_cache(maxsize=None)
def _m8_split_k_route_enabled() -> bool:
    """Route eight-token decode through the split-K M1 chain (ALPHAMOE_NVFP4_M8_SPLITK=1)."""
    import os

    return os.environ.get("ALPHAMOE_NVFP4_M8_SPLITK", "0") == "1"


def _alphamoe_complete_route_id(
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights_scale,
    topk_ids,
    out,
    top_k,
    block_m,
    w1_scale_prepared,
    w1_data_prepared=None,
    w1_gate_up_data_prepared=None,
    w1_gate_up_scale_prepared=None,
    w1_scale_prepared_interleaved=None,
    w2_scale_prepared=None,
    w2_data_prepared=None,
):
    """Select complete routes from immutable host metadata and owned sidecars.

    Route 9 (in-kernel-aligned prepared-W1 up + persistent BF16-route down) covers the eight-token
    decode shape and, when the caller also owns the interleaved W1 scale panels and the prepared W2
    scale panels its kernels read, the 9..``_ROUTE9_MAX_M``-token decode shapes as well.
    """
    m, packed_k = hidden_states.shape
    e, n, _ = gemm1_weights.shape
    k = packed_k * 2
    if (n, k, e, top_k, block_m) != (1024, 6144, 256, 8, 8):
        return 0
    if (m == 1 or (m == 8 and _m8_split_k_route_enabled())) and (
        is_alphamoe_nvfp4_small_alignment_seed_supported(
            topk_ids, e + 1, block_m, out, True
        )
    ):
        return 1
    if (
        m == 512
        and w1_gate_up_data_prepared is not None
        and w1_gate_up_scale_prepared is not None
    ):
        _check_prepared_w1_gate_up(
            gemm1_weights, w1_gate_up_data_prepared, w1_gate_up_scale_prepared
        )
        if (
            tuple(gemm1_weights_scale.shape) == (e, n, k // 16)
            and all(
                scale.data_ptr() % 4 == 0
                for scale in (
                    hidden_states_scale,
                    gemm1_weights_scale,
                    gemm2_weights_scale,
                )
            )
            and is_alphamoe_nvfp4_alignment_seed_supported(
                topk_ids, e + 1, block_m, out, True
            )
        ):
            return 14 if out.data_ptr() % 16 == 0 else 15
    if w1_scale_prepared is None:
        return 0
    prepared = (
        w1_scale_prepared.dtype == torch.uint8
        and w1_scale_prepared.is_contiguous()
        and w1_scale_prepared.device == gemm1_weights_scale.device
        and tuple(gemm1_weights_scale.shape) == (e, n, k // 16)
        and tuple(w1_scale_prepared.shape) == (e * (n // 128) * (k // 256), 16, 128)
        and w1_scale_prepared.data_ptr() % 16 == 0
        and all(
            scale.data_ptr() % 4 == 0
            for scale in (hidden_states_scale, gemm1_weights_scale, gemm2_weights_scale)
        )
    )
    if not prepared or not is_alphamoe_nvfp4_alignment_seed_supported(
        topk_ids, e + 1, block_m, out, True
    ):
        return 0
    if m == 8:
        return 9 if w1_data_prepared is not None else 2
    if (
        8 < m <= _ROUTE9_MAX_M
        and w1_data_prepared is not None
        and w1_scale_prepared_interleaved is not None
        and w2_scale_prepared is not None
    ):
        return 9
    if m == 128:
        if w1_data_prepared is not None and w2_data_prepared is not None:
            return 10 if out.data_ptr() % 16 == 0 else 11
        return 3
    if m == 512:
        if w1_data_prepared is not None and w2_data_prepared is not None:
            return 12 if out.data_ptr() % 16 == 0 else 13
        return 4 if out.data_ptr() % 16 == 0 else 5
    if m >= 128:
        return 6
    return 7 if m >= 8 else (8 if m >= 2 else 0)


def _alphamoe_try_complete_routed(
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    output1_scale_gate_scalar,
    output1_scale_scalar,
    output2_scale_scalar,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    topk_weights,
    out,
    topk_ids,
    cumsum_buffer,
    top_k,
    block_m,
    routed_scaling_factor,
    w1_scale_prepared,
    w1_data_prepared=None,
    w1_gate_up_data_prepared=None,
    w1_gate_up_scale_prepared=None,
    w2_scale_prepared=None,
    w1_scale_prepared_interleaved=None,
    w2_data_prepared=None,
    accumulate=True,
):
    route_id = _alphamoe_complete_route_id(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights_scale,
        topk_ids,
        out,
        top_k,
        block_m,
        w1_scale_prepared,
        w1_data_prepared,
        w1_gate_up_data_prepared,
        w1_gate_up_scale_prepared,
        w1_scale_prepared_interleaved,
        w2_scale_prepared,
        w2_data_prepared,
    )
    if route_id == 0:
        return False
    m, k = out.shape
    e, n, _ = gemm1_weights.shape
    capacity = expert_ids.numel()
    blocks = n // 256
    owner_plan = owner_count = initial_out = None
    partial_workspace = act_workspace = sf_workspace = None
    merge_counters = None
    if route_id == 1:
        initial_out = torch.empty_like(out, dtype=torch.float32)
        route_accumulator = torch.empty(
            (m * top_k, k), dtype=torch.float32, device=out.device
        )
        route_experts = torch.empty((m * top_k,), dtype=torch.int32, device=out.device)
        partial_workspace = torch.empty(
            (capacity * blocks, 8192), dtype=torch.float32, device=hidden_states.device
        )
        e_, n_, packed_k_ = gemm1_weights.shape
        if (
            w1_scale_prepared is not None
            and tuple(w1_scale_prepared.shape) == (e_ * (n_ // 128) * (packed_k_ // 128), 16, 128)
        ):
            # S5 one-token path: act/sf workspaces + zeroed self-resetting merge counters
            act_workspace = torch.empty(
                (capacity * blocks, 512), dtype=torch.uint8, device=hidden_states.device
            )
            sf_workspace = torch.empty(
                (capacity * blocks, 1024), dtype=torch.uint8, device=hidden_states.device
            )
            merge_counters = _s5_merge_counters(capacity * blocks, hidden_states.device)
    elif route_id == 8:
        if not accumulate:
            # accumulate=False hands over an uninitialized output. This route seeds its FP32
            # accumulator from ``out`` (``initial_out`` below) before the operator's own
            # accumulate=False zero fill runs, so clear the caller output here first.
            out.zero_()
        get_alphamoe_nvfp4_sm100_module().nvfp4_complete_small_alignment_op(
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output1_scale_gate_scalar,
            output1_scale_scalar,
            output2_scale_scalar,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            out,
            topk_ids,
            cumsum_buffer,
            top_k,
            block_m,
            routed_scaling_factor,
        )
        initial_out = out.to(torch.float32)
        route_accumulator = torch.zeros(
            (m * top_k, k), dtype=torch.float32, device=out.device
        )
        route_experts = torch.full(
            (m * top_k,), -1, dtype=torch.int32, device=out.device
        )
        act_workspace = torch.empty(
            (capacity * blocks, 512), dtype=torch.uint8, device=hidden_states.device
        )
        sf_workspace = torch.empty(
            (capacity * blocks, 1024), dtype=torch.uint8, device=hidden_states.device
        )
    else:
        owner_capacity = (topk_ids.numel() + 31) // 32 + e
        owner_plan = torch.empty(
            (owner_capacity, 3), dtype=torch.int32, device=hidden_states.device
        )
        owner_count = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
        if route_id in (2, 6, 7, 9):
            initial_out = torch.empty_like(out, dtype=torch.float32)
        act_workspace = torch.empty(
            (capacity * blocks, 512), dtype=torch.uint8, device=hidden_states.device
        )
        sf_workspace = torch.empty(
            (capacity * blocks, 1024), dtype=torch.uint8, device=hidden_states.device
        )
        route_dtype = (
            torch.bfloat16
            if route_id in (4, 5, 6, 9, 10, 11, 12, 13, 14, 15)
            else torch.float32
        )
        allocate_routes = torch.zeros if route_id == 7 else torch.empty
        route_accumulator = allocate_routes(
            (m * top_k, k), dtype=route_dtype, device=out.device
        )
        if route_id in (2, 3, 9, 10, 11, 12, 13, 14, 15):
            route_experts = torch.empty(
                (m * top_k,), dtype=torch.int32, device=out.device
            )
        else:
            route_experts = torch.full(
                (m * top_k,), -1, dtype=torch.int32, device=out.device
            )
    get_alphamoe_nvfp4_sm100_module().nvfp4_complete_routed_moe_op(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_gate_scalar,
        output1_scale_scalar,
        output2_scale_scalar,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        topk_ids,
        cumsum_buffer,
        route_accumulator,
        route_experts,
        owner_plan,
        owner_count,
        initial_out,
        partial_workspace,
        act_workspace,
        sf_workspace,
        top_k,
        block_m,
        routed_scaling_factor,
        w1_scale_prepared,
        route_id,
        w1_data_prepared,
        w1_gate_up_data_prepared,
        w1_gate_up_scale_prepared,
        w2_scale_prepared,
        w1_scale_prepared_interleaved,
        merge_counters,
        w2_data_prepared,
        accumulate,
    )
    return True


def _check_prepared_w1_gate_up(w1, data, scales):
    if data is None and scales is None:
        return
    if data is None or scales is None:
        raise ValueError(
            "Adjacent gate/up data and scale carriers must be supplied together"
        )
    e, n, packed_k = w1.shape
    records = e * (n // 256) * (packed_k // 128)
    for name, value, rows in (
        ("w1_gate_up_data_prepared", data, 256),
        ("w1_gate_up_scale_prepared", scales, 32),
    ):
        _require_cuda_tensor(name, value, dtype=torch.uint8, ndim=3)
        if (
            value.device != w1.device
            or not value.is_contiguous()
            or tuple(value.shape) != (records, rows, 128)
            or value.data_ptr() % 16 != 0
        ):
            raise ValueError(name + " must match the adjacent gate/up panel layout")


def prepare_nvfp4_w1_gate_up_data(w1):
    """Return byte-preserving adjacent gate/up panels for the routed entry.

    Contiguous uint8 W1 ``[E,N,K/2]`` requires ``N % 256 == 0`` and
    ``K % 256 == 0``. The result is ``[E*(N/256)*(K/256),256,128]``
    on the same device. Prepare after weight loading, before warmup or graph
    capture; keep raw weights and derived panels alive and immutable.
    """
    import torch

    assert w1.dtype == torch.uint8 and w1.ndim == 3 and w1.is_contiguous()
    experts, rows, packed_k = map(int, w1.shape)
    assert rows % 256 == 0 and packed_k % 128 == 0
    blocks, k_tiles = rows // 256, packed_k // 128
    return (
        w1.view(experts, 2, blocks, 128, k_tiles, 128)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(-1, 256, 128)
    )


def restore_nvfp4_w1_gate_up_data(prepared, original_shape):
    import torch

    experts, rows, packed_k = map(int, original_shape)
    assert rows % 256 == 0 and packed_k % 128 == 0
    blocks, k_tiles = rows // 256, packed_k // 128
    assert prepared.dtype == torch.uint8 and prepared.is_contiguous()
    assert tuple(prepared.shape) == (experts * blocks * k_tiles, 256, 128)
    return (
        prepared.view(experts, blocks, k_tiles, 2, 128, 128)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(experts, rows, packed_k)
    )


def prepare_nvfp4_w1_gate_up_scales(prepared_scales, original_weight_shape):
    """Pair existing prepared W1 scale bytes with adjacent gate/up data.

    ``prepared_scales`` is the uint8 output of :func:`prepare_nvfp4_w1_scales`;
    ``original_weight_shape`` is the raw W1 shape ``[E,N,K/2]``. The result
    is ``[E*(N/256)*(K/256),32,128]`` and preserves every E4M3 byte.
    Reuse it with data prepared from the same device-local weight load.
    """
    import torch

    experts, rows, packed_k = map(int, original_weight_shape)
    assert rows % 256 == 0 and packed_k % 128 == 0
    blocks, k_tiles = rows // 256, packed_k // 128
    assert prepared_scales.dtype == torch.uint8 and prepared_scales.is_contiguous()
    assert tuple(prepared_scales.shape) == (experts * 2 * blocks * k_tiles, 16, 128)
    return (
        prepared_scales.view(experts, 2, blocks, k_tiles, 16, 128)
        .permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .view(-1, 32, 128)
    )


def restore_nvfp4_w1_gate_up_scales(prepared, original_weight_shape):
    import torch

    experts, rows, packed_k = map(int, original_weight_shape)
    assert rows % 256 == 0 and packed_k % 128 == 0
    blocks, k_tiles = rows // 256, packed_k // 128
    assert prepared.dtype == torch.uint8 and prepared.is_contiguous()
    assert tuple(prepared.shape) == (experts * blocks * k_tiles, 32, 128)
    return (
        prepared.view(experts, blocks, k_tiles, 2, 16, 128)
        .permute(0, 3, 1, 2, 4, 5)
        .contiguous()
        .view(-1, 16, 128)
    )


# Deferred finalize: the routed decode chain runs without the caller's seed and
# the caller finalizes later with a BF16 seed (for example a shared-expert output).
__all__ += [
    "AlphaMoeNvfp4DeferredOutput",
    "alphamoe_nvfp4_routed_moe_deferred",
    "alphamoe_nvfp4_finalize_deferred",
]

_DEFERRABLE_COMPLETE_ROUTES = (1, 2, 7, 8, 9)


class AlphaMoeNvfp4DeferredOutput:
    """Routed expert contributions awaiting :func:`alphamoe_nvfp4_finalize_deferred`.

    For the deferrable complete routes the per-route accumulators, route experts
    and top-k weights are held until the caller supplies its BF16 seed. For every
    other route the weighted expert sum has already been written into ``out``
    from a zero seed and the finalize step adds it to the caller's seed.
    """

    __slots__ = (
        "route_id",
        "route_accumulator",
        "route_experts",
        "topk_weights",
        "output2_scale_scalar",
        "routed_scaling_factor",
        "top_k",
        "out",
    )

    def __init__(
        self,
        route_id: int,
        route_accumulator: Optional[torch.Tensor],
        route_experts: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        routed_scaling_factor: float,
        top_k: int,
        out: Optional[torch.Tensor],
    ) -> None:
        self.route_id = int(route_id)
        self.route_accumulator = route_accumulator
        self.route_experts = route_experts
        self.topk_weights = topk_weights
        self.output2_scale_scalar = output2_scale_scalar
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.top_k = int(top_k)
        self.out = out

    @property
    def deferred(self) -> bool:
        return self.out is None


def alphamoe_nvfp4_routed_moe_deferred(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    topk_ids: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    top_k: int,
    block_m: int = 8,
    routed_scaling_factor: float = 1.0,
    w1_scale_prepared: Optional[torch.Tensor] = None,
    w2_scale_prepared: Optional[torch.Tensor] = None,
    w1_data_prepared: Optional[torch.Tensor] = None,
    w1_gate_up_data_prepared: Optional[torch.Tensor] = None,
    w1_gate_up_scale_prepared: Optional[torch.Tensor] = None,
    w1_scale_prepared_interleaved: Optional[torch.Tensor] = None,
    w2_data_prepared: Optional[torch.Tensor] = None,
) -> AlphaMoeNvfp4DeferredOutput:
    """Run alignment and the expert GEMMs now; finalize later with a BF16 seed.

    Same inputs and route selection as :func:`alphamoe_nvfp4_routed_moe`. ``out``
    is the caller's BF16 ``[M, K]`` output workspace: its contents are ignored.
    For the deferrable complete routes (decode shapes M=1, 2..7, 8 and 9..127)
    nothing is written to ``out`` here; the seed conversion and the finalize
    launch are skipped and the private route accumulators are returned. For
    every other route ``out`` is zero-filled and receives the finished weighted
    expert sum exactly as the existing routed entry produces it from a zero seed.
    Call :func:`alphamoe_nvfp4_finalize_deferred` with the caller's BF16 seed to
    obtain ``seed + routed contributions`` in either case. All launches use the
    current stream, including during CUDA graph capture.
    """
    _check_alphamoe_nvfp4_supported(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_gate_scalar,
        output1_scale_scalar,
        output2_scale_scalar,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        routed_scaling_factor,
        w1_scale_prepared,
        w2_scale_prepared,
    )
    _check_prepared_w1_data(gemm1_weights, w1_data_prepared)
    _check_prepared_w2_data(gemm2_weights, w2_data_prepared)
    _check_prepared_w1_gate_up(
        gemm1_weights, w1_gate_up_data_prepared, w1_gate_up_scale_prepared
    )
    _check_prepared_w1_interleaved(
        gemm1_weights, gemm1_weights_scale, w1_scale_prepared_interleaved
    )
    route_id = _alphamoe_complete_route_id(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights_scale,
        topk_ids,
        out,
        top_k,
        block_m,
        w1_scale_prepared,
        w1_data_prepared,
        w1_gate_up_data_prepared,
        w1_gate_up_scale_prepared,
        w1_scale_prepared_interleaved,
        w2_scale_prepared,
        w2_data_prepared,
    )
    topk_weights = topk_weights.contiguous()
    if route_id not in _DEFERRABLE_COMPLETE_ROUTES:
        out.zero_()
        alphamoe_nvfp4_routed_moe(
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output1_scale_gate_scalar,
            output1_scale_scalar,
            output2_scale_scalar,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            out,
            topk_ids,
            cumsum_buffer,
            top_k,
            block_m,
            routed_scaling_factor,
            w1_scale_prepared,
            w2_scale_prepared,
            w1_data_prepared,
            w1_gate_up_data_prepared,
            w1_gate_up_scale_prepared,
            w1_scale_prepared_interleaved,
            w2_data_prepared=w2_data_prepared,
        )
        return AlphaMoeNvfp4DeferredOutput(
            route_id, None, None, topk_weights, output2_scale_scalar,
            routed_scaling_factor, top_k, out,
        )
    m, k = out.shape
    e, n, _ = gemm1_weights.shape
    capacity = expert_ids.numel()
    blocks = n // 256
    module = get_alphamoe_nvfp4_sm100_module()
    owner_plan = owner_count = None
    partial_workspace = act_workspace = sf_workspace = None
    merge_counters = None
    if route_id == 1:
        route_accumulator = torch.empty(
            (m * top_k, k), dtype=torch.float32, device=out.device
        )
        route_experts = torch.empty((m * top_k,), dtype=torch.int32, device=out.device)
        partial_workspace = torch.empty(
            (capacity * blocks, 8192), dtype=torch.float32, device=hidden_states.device
        )
        e_, n_, packed_k_ = gemm1_weights.shape
        if (
            w1_scale_prepared is not None
            and tuple(w1_scale_prepared.shape) == (e_ * (n_ // 128) * (packed_k_ // 128), 16, 128)
        ):
            # S5 one-token path: act/sf workspaces + zeroed self-resetting merge counters
            act_workspace = torch.empty(
                (capacity * blocks, 512), dtype=torch.uint8, device=hidden_states.device
            )
            sf_workspace = torch.empty(
                (capacity * blocks, 1024), dtype=torch.uint8, device=hidden_states.device
            )
            merge_counters = _s5_merge_counters(capacity * blocks, hidden_states.device)
    elif route_id == 8:
        module.nvfp4_complete_small_alignment_op(
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            output1_scale_gate_scalar,
            output1_scale_scalar,
            output2_scale_scalar,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            out,
            topk_ids,
            cumsum_buffer,
            top_k,
            block_m,
            routed_scaling_factor,
        )
        route_accumulator = torch.zeros(
            (m * top_k, k), dtype=torch.float32, device=out.device
        )
        route_experts = torch.full(
            (m * top_k,), -1, dtype=torch.int32, device=out.device
        )
        act_workspace = torch.empty(
            (capacity * blocks, 512), dtype=torch.uint8, device=hidden_states.device
        )
        sf_workspace = torch.empty(
            (capacity * blocks, 1024), dtype=torch.uint8, device=hidden_states.device
        )
    else:
        owner_capacity = (topk_ids.numel() + 31) // 32 + e
        owner_plan = torch.empty(
            (owner_capacity, 3), dtype=torch.int32, device=hidden_states.device
        )
        owner_count = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
        act_workspace = torch.empty(
            (capacity * blocks, 512), dtype=torch.uint8, device=hidden_states.device
        )
        sf_workspace = torch.empty(
            (capacity * blocks, 1024), dtype=torch.uint8, device=hidden_states.device
        )
        route_dtype = torch.bfloat16 if route_id == 9 else torch.float32
        allocate_routes = torch.zeros if route_id == 7 else torch.empty
        route_accumulator = allocate_routes(
            (m * top_k, k), dtype=route_dtype, device=out.device
        )
        if route_id in (2, 9):
            route_experts = torch.empty(
                (m * top_k,), dtype=torch.int32, device=out.device
            )
        else:
            route_experts = torch.full(
                (m * top_k,), -1, dtype=torch.int32, device=out.device
            )
    module.nvfp4_complete_routed_deferred_moe_op(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_gate_scalar,
        output1_scale_scalar,
        output2_scale_scalar,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        topk_ids,
        cumsum_buffer,
        route_accumulator,
        route_experts,
        owner_plan,
        owner_count,
        None,
        partial_workspace,
        act_workspace,
        sf_workspace,
        top_k,
        block_m,
        routed_scaling_factor,
        w1_scale_prepared,
        route_id,
        w1_data_prepared,
        w1_gate_up_data_prepared,
        w1_gate_up_scale_prepared,
        w2_scale_prepared,
        w1_scale_prepared_interleaved,
        merge_counters,
        w2_data_prepared,
    )
    return AlphaMoeNvfp4DeferredOutput(
        route_id, route_accumulator, route_experts, topk_weights,
        output2_scale_scalar, routed_scaling_factor, top_k, None,
    )


def alphamoe_nvfp4_finalize_deferred(
    deferred: AlphaMoeNvfp4DeferredOutput,
    seed: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Write ``seed + routed contributions`` for a deferred routed-MoE call.

    ``seed`` is the caller's contiguous BF16 ``[M, K]`` tensor (for example the
    shared-expert output). ``out`` defaults to ``seed`` (in place); a distinct
    ``out`` must not overlap ``seed``. For deferrable routes this launches the
    route's finalize kernel with the BF16 seed under programmatic dependent
    launch; the arithmetic equals the existing routed entry given the same
    seed. For other routes the finished expert sum is added into ``out``.
    Returns ``out``.
    """
    _require_cuda_tensor("seed", seed, dtype=torch.bfloat16, ndim=2)
    if out is None:
        out = seed
    else:
        _require_cuda_tensor("out", out, dtype=torch.bfloat16, ndim=2)
        if out.shape != seed.shape or out.device != seed.device:
            raise ValueError("out must match the seed shape and device")
    if not (seed.is_contiguous() and out.is_contiguous()):
        raise ValueError("seed and out must be contiguous")
    if not deferred.deferred:
        if deferred.out.shape != seed.shape:
            raise ValueError("deferred output does not match the seed shape")
        if out.data_ptr() == seed.data_ptr():
            out.add_(deferred.out)
        else:
            torch.add(seed, deferred.out, out=out)
        return out
    get_alphamoe_nvfp4_sm100_module().nvfp4_complete_routed_finalize_op(
        deferred.route_accumulator,
        deferred.route_experts,
        deferred.output2_scale_scalar,
        deferred.topk_weights,
        seed,
        out,
        deferred.top_k,
        deferred.routed_scaling_factor,
        deferred.route_id,
    )
    return out
