"""
Copyright (c) 2025 by FlashInfer team.

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
import warnings
from enum import IntEnum
from typing import Dict, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass import torch as cutlass_torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from ...jit.moe_utils import gen_moe_utils_module
from ...tllm_enums import ActivationType, is_gated_activation, normalize_activation_type
from ...utils import get_compute_capability


def _get_cuda_stream_ptr() -> int:
    """Get the current PyTorch CUDA stream pointer.

    This is needed for CUDA graph compatibility - the kernel must run on
    PyTorch's current stream, not TVM's default stream.
    """
    return torch.cuda.current_stream().cuda_stream


# ============================ Helper Functions ============================


SUPPORTED_CUTE_DSL_MOE_ACTIVATION_TYPES = (
    ActivationType.Swiglu,
    ActivationType.GegluTanh,
    ActivationType.Relu2,
)


def normalize_cute_dsl_moe_activation_type(
    activation_type: Union[int, ActivationType],
) -> Tuple[ActivationType, bool]:
    activation_type = normalize_activation_type(activation_type)
    if activation_type not in SUPPORTED_CUTE_DSL_MOE_ACTIVATION_TYPES:
        expected = " or ".join(repr(t) for t in SUPPORTED_CUTE_DSL_MOE_ACTIVATION_TYPES)
        raise ValueError(
            f"Unsupported activation_type {activation_type!r}; expected {expected}"
        )
    return activation_type, is_gated_activation(activation_type)


def validate_cute_dsl_moe_situ_config(
    activation_type: ActivationType,
    situ_beta: Optional[float],
    situ_linear_beta: Optional[float],
) -> None:
    """Validate the optional SiTU variant of the SwiGLU epilogue."""
    if situ_beta is None:
        if situ_linear_beta is not None:
            raise ValueError("situ_linear_beta requires situ_beta")
        return
    if activation_type != ActivationType.Swiglu:
        raise ValueError("SiTU parameters require ActivationType.Swiglu")
    if not math.isfinite(situ_beta) or situ_beta <= 0:
        raise ValueError("situ_beta must be positive and finite")
    if situ_linear_beta is not None and (
        not math.isfinite(situ_linear_beta) or situ_linear_beta <= 0
    ):
        raise ValueError("situ_linear_beta must be positive and finite when set")


def normalize_cute_dsl_moe_weight_interleave(
    weight_interleave: Optional[int], swap_ab: bool
) -> int:
    """Validate and resolve the physical up/gate weight interleave."""
    if weight_interleave is None:
        weight_interleave = 16 if swap_ab else 64
    valid_values = (16,) if swap_ab else (16, 64)
    if (
        isinstance(weight_interleave, bool)
        or not isinstance(weight_interleave, int)
        or weight_interleave not in valid_values
    ):
        raise ValueError(
            f"weight_interleave must be one of {valid_values} "
            f"when swap_ab={swap_ab}, got {weight_interleave!r}"
        )
    return weight_interleave


def warn_deprecated_cute_dsl_moe_weight_interleave(
    weight_interleave: int, device: torch.device
) -> None:
    """Warn for the legacy Blackwell up/gate weight layout."""
    device = torch.device(device)
    if (
        weight_interleave == 64
        and device.type == "cuda"
        and torch.cuda.is_available()
        and get_compute_capability(device) in ((10, 0), (10, 3))
    ):
        warnings.warn(
            "weight_interleave=64 is deprecated for W4A4/W4A8 CuTe-DSL MoE "
            "on SM100/SM103; use weight_interleave=16 instead.",
            DeprecationWarning,
            stacklevel=3,
        )


def get_max_num_tiles(
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    tile_size: int,
) -> int:
    """
    Calculate the tight upper bound on the number of tiles produced by
    moe_sort for a given (num_tokens, top_k, num_local_experts, tile_size).

    Mirrors TRT-LLM's ``GroupedGemmInputsHelper.get_max_num_tiles()`` in
    ``tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py``. The compact
    closed-form expression is the tight worst-case bound on
    ``sum_e ceil(K_e / tile_size)`` subject to ``sum_e K_e = E`` (where E
    = num_tokens * top_k and K_e is the per-local-expert token count).

    The worst case is achieved when (L-1) experts each have exactly 1
    token (each contributing 1 fully-padded tile) and one expert has the
    remaining ``E - L + 1`` tokens. Using the identity
    ``ceil((X+1)/T) = floor(X/T) + 1`` (valid for non-negative integer X),
    that worst case simplifies to ``L + floor((E - L) / T)``, which is
    algebraically equal to ``(E + (T - 1) * L) // T``.

    Args:
        num_tokens: Number of input tokens.
        top_k: Number of experts per token.
        num_local_experts: Number of local experts (for expert parallelism).
        tile_size: Tile size for scheduling (moe_sort's mPaddingLog2 /
            mTileTokensDim).

    Returns:
        Maximum number of tiles. Sized to fit any routing distribution
        of ``num_tokens * top_k`` expanded tokens across ``num_local_experts``
        local experts.
    """
    num_expanded_tokens = num_tokens * top_k

    if num_expanded_tokens <= num_local_experts:
        return num_expanded_tokens

    return (num_expanded_tokens + (tile_size - 1) * num_local_experts) // tile_size


def get_token_capacity(num_tokens: int) -> int:
    """Return the power-of-two launcher capacity for a token count."""
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    return 1 << (num_tokens - 1).bit_length()


def get_max_num_permuted_tokens(
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    tile_size: int,
) -> int:
    """
    Calculate the maximum number of permuted tokens.

    This follows the same logic as TRT-LLM's GroupedGemmInputsHelper.get_max_num_permuted_tokens().

    Args:
        num_tokens: Number of input tokens.
        top_k: Number of experts per token.
        num_local_experts: Number of local experts (for expert parallelism).
        tile_size: Tile size for scheduling.

    Returns:
        Maximum number of permuted tokens.
    """
    max_num_tiles = get_max_num_tiles(num_tokens, top_k, num_local_experts, tile_size)
    return max_num_tiles * tile_size


class MoeActivationType(IntEnum):
    """Activation types for MoE layers.

    Note: Must match MoeActivationType enum in moeUtils.h
    """

    Gelu = 0
    Relu = 1
    Silu = 2
    Swiglu = 3
    Geglu = 4
    Identity = 5


def _cutlass_type(dtype: torch.dtype):
    """Resolve a torch dtype through CUTLASS's canonical dtype mapping."""
    for cutlass_dtype in (cutlass.BFloat16, cutlass.Float16, cutlass.Float32):
        if cutlass_torch.dtype(cutlass_dtype) == dtype:
            return cutlass_dtype
    raise ValueError(f"unsupported routing dtype: {dtype}")


@functools.lru_cache(maxsize=1)
def _get_moe_utils_module():
    """Lazily load and cache the MoE utils JIT module."""
    spec = gen_moe_utils_module()
    return spec.build_and_load()


def _get_dtype_suffix(dtype: torch.dtype) -> str:
    """Get the dtype suffix for function dispatch."""
    if dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float8_e4m3fn:
        return "fp8"
    elif dtype == torch.uint8:  # Used for FP4 (packed)
        return "fp4"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def moe_permute(
    input: torch.Tensor,
    permuted_output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    max_num_permuted_tokens: int,
    top_k: int,
    tile_size: int,
    enable_pdl: bool = False,
    input_sf: Optional[torch.Tensor] = None,
    permuted_sf: Optional[torch.Tensor] = None,
) -> None:
    """
    Permute input activations according to MoE routing decisions.

    This function reorders input tokens based on expert assignments, preparing
    them for batched expert computation.

    Args:
        input: Input activations tensor of shape [num_tokens, hidden_size].
               Supported dtypes: float16, bfloat16, float8_e4m3fn, uint8 (FP4).
        permuted_output: Output tensor for permuted activations of shape
                        [max_num_permuted_tokens, hidden_size].
        tile_idx_to_mn_limit: Tensor mapping tile indices to M/N limits.
                             Shape: [num_tiles].
        permuted_idx_to_expanded_idx: Mapping from permuted indices to expanded indices.
                                      Shape: [max_num_permuted_tokens].
        num_non_exiting_tiles: Number of non-exiting tiles (scalar on device).
        max_num_permuted_tokens: Maximum number of permuted tokens.
        top_k: Number of experts per token.
        tile_size: Size of each tile for scheduling.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
        input_sf: Scale factors for input (required for FP4).
                  Shape: [num_tokens, hidden_size // 16].
        permuted_sf: Output scale factors for permuted data (required for FP4).
                     Shape: [max_num_permuted_tokens, hidden_size // 16].

    Note:
        - For FP4 inputs, input_sf and permuted_sf are required.
        - The permuted_sf output uses a swizzled layout for efficient TMA access.
    """
    module = _get_moe_utils_module()
    dtype_suffix = _get_dtype_suffix(input.dtype)

    hidden_size = input.shape[-1]
    if dtype_suffix == "fp4":
        # For FP4, hidden_size is halved due to packing
        hidden_size = hidden_size * 2

    func_name = f"flashinfer_moe_permute_{dtype_suffix}"
    func = module[func_name]

    input_sf_ptr = input_sf.data_ptr() if input_sf is not None else 0
    permuted_sf_ptr = permuted_sf.data_ptr() if permuted_sf is not None else 0

    func(
        input.data_ptr(),
        permuted_output.data_ptr(),
        input_sf_ptr,
        permuted_sf_ptr,
        tile_idx_to_mn_limit.data_ptr(),
        permuted_idx_to_expanded_idx.data_ptr(),
        num_non_exiting_tiles.data_ptr(),
        max_num_permuted_tokens,
        hidden_size,
        top_k,
        tile_size,
        enable_pdl,
        _get_cuda_stream_ptr(),
    )


def moe_unpermute(
    permuted_input: torch.Tensor,
    output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    topk_scales: torch.Tensor,
    num_tokens: int,
    top_k: int,
    enable_pdl: bool = False,
    input_is_expanded: bool = False,
) -> None:
    """
    Unpermute and scale outputs after expert computation.

    This function reverses the permutation done by moe_permute and applies
    top-k scaling weights to combine expert outputs.

    Args:
        permuted_input: Permuted expert outputs of shape [num_permuted_tokens, hidden_size].
                        Supported dtypes: float16, bfloat16.
        output: Output tensor of shape [num_tokens, hidden_size].
        expanded_idx_to_permuted_idx: Mapping from expanded indices to permuted indices.
                                       Shape: [num_tokens, top_k].
                                       -1 indicates a masked expert.
        topk_scales: Scaling weights for each expert per token.
                     Shape: [num_tokens, top_k].
                     Supported dtypes: float32, float16, bfloat16.
        num_tokens: Number of original tokens.
        top_k: Number of experts per token.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
        input_is_expanded: Whether input rows use expanded (token, top-k slot)
            order instead of expert-permuted order.

    Note:
        Output is the weighted sum of expert contributions:
        output[i] = sum(topk_scales[i, k] * expert_output[i, k] for k in range(top_k))
    """
    module = _get_moe_utils_module()
    input_dtype_suffix = _get_dtype_suffix(permuted_input.dtype)

    hidden_size = permuted_input.shape[-1]

    # Determine scale dtype suffix
    if topk_scales.dtype == torch.float32:
        scale_suffix = "float"
    elif topk_scales.dtype == torch.float16:
        scale_suffix = "half"
    elif topk_scales.dtype == torch.bfloat16:
        scale_suffix = "bf16"
    else:
        raise ValueError(f"Unsupported scale dtype: {topk_scales.dtype}")

    func_name = f"flashinfer_moe_unpermute_{input_dtype_suffix}_{scale_suffix}_scale"
    func = module[func_name]

    func(
        permuted_input.data_ptr(),
        output.data_ptr(),
        expanded_idx_to_permuted_idx.data_ptr(),
        topk_scales.data_ptr(),
        num_tokens,
        hidden_size,
        top_k,
        input_is_expanded,
        enable_pdl,
        _get_cuda_stream_ptr(),
    )


def moe_output_memset(
    output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    max_num_permuted_tokens: int,
    top_k: int,
    tile_size: int,
    enable_pdl: bool = False,
) -> None:
    """
    Zero-initialize output buffer for tokens that will receive scattered writes.

    This function sets output locations to zero for tokens that are first in their
    top-k sequence, preparing the buffer for accumulation during unpermutation.

    Args:
        output: Output tensor to zero-initialize. Shape: [num_tokens, hidden_size].
                Supported dtypes: float16, bfloat16.
        tile_idx_to_mn_limit: Tensor mapping tile indices to M/N limits.
                             Shape: [num_tiles].
        expanded_idx_to_permuted_idx: Mapping from expanded indices to permuted indices.
                                       Shape: [num_tokens, top_k].
        permuted_idx_to_expanded_idx: Mapping from permuted indices to expanded indices.
                                      Shape: [max_num_permuted_tokens].
        num_non_exiting_tiles: Number of non-exiting tiles (scalar on device).
        max_num_permuted_tokens: Maximum number of permuted tokens.
        top_k: Number of experts per token.
        tile_size: Size of each tile for scheduling.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
    """
    module = _get_moe_utils_module()
    dtype_suffix = _get_dtype_suffix(output.dtype)

    hidden_size = output.shape[-1]

    func_name = f"flashinfer_moe_output_memset_{dtype_suffix}"
    func = module[func_name]

    func(
        output.data_ptr(),
        tile_idx_to_mn_limit.data_ptr(),
        expanded_idx_to_permuted_idx.data_ptr(),
        permuted_idx_to_expanded_idx.data_ptr(),
        num_non_exiting_tiles.data_ptr(),
        max_num_permuted_tokens,
        hidden_size,
        top_k,
        tile_size,
        enable_pdl,
    )


def moe_output_memset_inplace(output: torch.Tensor) -> None:
    """
    Zero the active MoE output slice via ``cudaMemsetAsync`` on the current
    CUDA stream.

    Dense-only port of TRT-LLM's
    ``torch.ops.trtllm.moe_output_memset_inplace`` Path A
    (``cuteDslMoeUtilsOp.cpp:moe_output_memset_inplace`` at the
    ``!enable_alltoall || ep_size <= top_k`` branch). Functionally
    equivalent to ``output.zero_()`` but with lower per-call launch overhead
    (one ``cudaMemsetAsync`` vs PyTorch's ``FillFunctor`` kernel launch —
    saves ~2-3 µs per call at the cells where memset cost is visible).

    This entry point exposes only Path A. Current callers of the
    monolithic CuteDSL MoE API handle all-to-all outside this function,
    so TRT-LLM's internal-alltoall Path B (the sparse
    ``moeOutputMemset`` kernel) is not part of this API. The existing
    sparse ``moe_output_memset`` binding remains available if a future
    internal-alltoall integration needs it.

    The wrapper passes PyTorch's current CUDA stream pointer explicitly
    to the C++ binding (via ``_get_cuda_stream_ptr()``). This is
    required because the underlying ``get_current_stream()`` C++ helper
    resolves through ``TVMFFIEnvGetStream``, which does NOT track
    PyTorch's ``torch.cuda.stream(...)`` Python context — without the
    explicit pointer the memset would queue on TVM's env stream and
    would not overlap aux-stream memset with surrounding GEMM work.
    Same pattern as ``moe_sort`` in this file.

    Args:
        output: Output tensor to zero. Shape: ``[num_tokens, hidden_size]``.
                Supported dtypes: ``torch.float16``, ``torch.bfloat16``.
    """
    if output.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "moe_output_memset_inplace only supports torch.float16 and "
            f"torch.bfloat16, got {output.dtype}"
        )
    if output.dim() != 2:
        raise ValueError(
            "moe_output_memset_inplace expects a 2D tensor, "
            f"got shape {tuple(output.shape)}"
        )
    if not output.is_contiguous():
        raise ValueError(
            "moe_output_memset_inplace requires a contiguous tensor; "
            "cudaMemsetAsync zeros a dense byte range from data_ptr()"
        )

    module = _get_moe_utils_module()
    dtype_suffix = _get_dtype_suffix(output.dtype)

    num_tokens, hidden_size = output.shape

    func_name = f"flashinfer_moe_output_memset_inplace_{dtype_suffix}"
    func = module[func_name]

    func(output.data_ptr(), num_tokens, hidden_size, _get_cuda_stream_ptr())


# ============================ moe_sort ============================


def allocate_moe_sort_buffers(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    tile_tokens_dim: int = 128,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Pre-allocate output buffers for moe_sort for CUDA graph compatibility.

    When using CUDA graphs, allocate these buffers BEFORE graph capture and pass
    them to moe_sort via the out_* parameters. This ensures the same memory
    addresses are used during capture and replay.

    Args:
        num_tokens: Number of tokens.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        num_local_experts: Number of local experts. Default: num_experts.
        tile_tokens_dim: Tile size for scheduling. Default: 128.
        device: Device to allocate on. Default: "cuda".

    Returns:
        Dictionary with pre-allocated buffers that can be unpacked as kwargs to moe_sort:
            - out_tile_idx_to_expert_idx
            - out_tile_idx_to_mn_limit
            - out_expanded_idx_to_permuted_idx
            - out_permuted_idx_to_expanded_idx
            - out_total_num_padded_tokens
            - out_num_non_exiting_tiles

    Example:
        >>> # Pre-allocate before CUDA graph capture
        >>> buffers = allocate_moe_sort_buffers(num_tokens, num_experts, top_k)
        >>>
        >>> # Warmup
        >>> for _ in range(3):
        ...     moe_sort(experts, scales, ..., **buffers)
        >>>
        >>> # Capture
        >>> g = torch.cuda.CUDAGraph()
        >>> with torch.cuda.graph(g):
        ...     results = moe_sort(experts, scales, ..., **buffers)
    """
    from .blackwell.moe_sort import MoeSortKernel

    if num_local_experts is None:
        num_local_experts = num_experts

    max_num_tiles = get_max_num_tiles(
        num_tokens, top_k, num_local_experts, tile_tokens_dim
    )
    max_num_permuted_tokens = get_max_num_permuted_tokens(
        num_tokens, top_k, num_local_experts, tile_tokens_dim
    )
    with torch.cuda.device(device):
        num_ctas = MoeSortKernel(
            num_tokens,
            num_experts,
            top_k,
            0,
            num_local_experts,
            tile_tokens_dim,
            use_pdl=False,
        ).num_ctas

    return {
        "out_tile_idx_to_expert_idx": torch.empty(
            (max_num_tiles,), dtype=torch.int32, device=device
        ),
        "out_tile_idx_to_mn_limit": torch.empty(
            (max_num_tiles,), dtype=torch.int32, device=device
        ),
        "out_expanded_idx_to_permuted_idx": torch.empty(
            (num_tokens, top_k), dtype=torch.int32, device=device
        ),
        "out_permuted_idx_to_expanded_idx": torch.empty(
            (max_num_permuted_tokens,), dtype=torch.int32, device=device
        ),
        "out_total_num_padded_tokens": torch.empty(
            (1,), dtype=torch.int32, device=device
        ),
        "out_num_non_exiting_tiles": torch.empty(
            (1,), dtype=torch.int32, device=device
        ),
        "global_counts": torch.zeros(
            (2 * num_local_experts,), dtype=torch.int32, device=device
        ),
        "global_offsets": torch.empty(
            (num_ctas * num_local_experts,), dtype=torch.int32, device=device
        ),
        "grid_sync": torch.zeros((4,), dtype=torch.int32, device=device),
    }


def allocate_moe_routing_buffers(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device | str,
    padded_m: Optional[int] = None,
    tile_size: Optional[int] = None,
    capacity: Optional[int] = None,
    emit_expanded_to_permuted: bool = False,
) -> Dict[str, torch.Tensor]:
    """Pre-allocate outputs and scratch buffers for ``moe_routing``."""
    from .blackwell.moe_routing import MoeRoutingKernel

    if capacity is not None and emit_expanded_to_permuted:
        raise ValueError(
            "fixed-slot routing already emits the expanded-to-slot mapping"
        )
    device = torch.device(device)
    max_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    kernel = MoeRoutingKernel(
        num_tokens,
        num_experts,
        top_k,
        padded_m=padded_m,
        tile_size=tile_size,
        capacity=capacity,
        use_pdl=False,
        emit_expanded_to_permuted=emit_expanded_to_permuted,
        max_ctas=max_ctas,
    )
    fixed_slot = capacity is not None
    if fixed_slot:
        num_slots = num_experts * int(capacity)
        output0 = torch.empty(num_slots, dtype=torch.int64, device=device)
        output1 = torch.empty(num_experts, dtype=torch.int32, device=device)
        output2 = torch.empty(num_slots, dtype=torch.int32, device=device)
        output3 = torch.empty(num_tokens * top_k, dtype=torch.int32, device=device)
    else:
        if padded_m is None or tile_size is None:
            raise ValueError("tiled routing requires padded_m and tile_size")
        num_tiles = padded_m // tile_size
        output0 = torch.empty(num_tiles, dtype=torch.int32, device=device)
        output1 = torch.empty_like(output0)
        output2 = torch.empty(padded_m, dtype=torch.int32, device=device)
        output3 = torch.empty(1, dtype=torch.int32, device=device)
    return {
        "token_final_scales": torch.empty(
            num_tokens, top_k, dtype=torch.float32, device=device
        ),
        "token_selected_experts": torch.empty(
            num_tokens, top_k, dtype=torch.int32, device=device
        ),
        "output0": output0,
        "output1": output1,
        "output2": output2,
        "output3": output3,
        "expanded_to_permuted": torch.empty(
            (num_tokens, top_k) if emit_expanded_to_permuted else (1,),
            dtype=torch.int32,
            device=device,
        ),
        "global_counts": torch.zeros(
            (
                2 * num_experts
                if not fixed_slot and num_tokens > 2048
                else kernel.num_ctas * num_experts
            ),
            dtype=torch.int32,
            device=device,
        ),
        "global_cursors": torch.empty(
            kernel.num_ctas * num_experts, dtype=torch.int32, device=device
        ),
        "global_routed_experts": torch.empty(
            num_tokens * top_k, dtype=torch.int32, device=device
        ),
        "grid_sync": torch.zeros(2, dtype=torch.int32, device=device),
    }


def moe_routing(
    scores: torch.Tensor,
    top_k: int,
    padded_m: Optional[int] = None,
    tile_size: Optional[int] = None,
    capacity: Optional[int] = None,
    use_pdl: bool = True,
    emit_expanded_to_permuted: bool = False,
    token_capacity: Optional[int] = None,
    routing_buffers: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """Select top-k experts and build grouped-GEMM routing mappings."""
    if scores.ndim != 2 or not scores.is_contiguous():
        raise ValueError("scores must be a contiguous 2D tensor")
    if scores.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("scores must have dtype bf16, fp16, or fp32")
    if capacity is not None and emit_expanded_to_permuted:
        raise ValueError(
            "fixed-slot routing already emits the expanded-to-slot mapping"
        )
    num_tokens, num_experts = scores.shape
    token_capacity = token_capacity or num_tokens
    if not 0 < num_tokens <= token_capacity:
        raise ValueError("token_capacity must cover scores")
    if capacity is not None and num_tokens != token_capacity:
        raise ValueError("fixed-slot routing requires an exact token capacity")
    routing_buffers = routing_buffers or allocate_moe_routing_buffers(
        token_capacity,
        num_experts,
        top_k,
        scores.device,
        padded_m,
        tile_size,
        capacity,
        emit_expanded_to_permuted,
    )
    compiled = compile_moe_routing(
        token_capacity,
        num_experts,
        top_k,
        padded_m,
        tile_size,
        capacity,
        use_pdl,
        emit_expanded_to_permuted,
        scores.dtype,
        scores.device,
    )
    token_final_scales = routing_buffers["token_final_scales"][:num_tokens]
    token_selected_experts = routing_buffers["token_selected_experts"][:num_tokens]
    expanded_to_permuted = routing_buffers["expanded_to_permuted"]
    if emit_expanded_to_permuted:
        expanded_to_permuted = expanded_to_permuted[:num_tokens]
    compiled(
        scores.detach(),
        token_final_scales,
        token_selected_experts,
        routing_buffers["output0"],
        routing_buffers["output1"],
        routing_buffers["output2"],
        routing_buffers["output3"],
        expanded_to_permuted,
        routing_buffers["global_counts"],
        routing_buffers["global_cursors"],
        routing_buffers["global_routed_experts"],
        routing_buffers["grid_sync"],
        cuda.CUstream(_get_cuda_stream_ptr()),
    )
    if capacity is not None:
        return {
            "token_final_scales": token_final_scales,
            "token_selected_experts": token_selected_experts,
            "token_ids": routing_buffers["output0"],
            "expert_counts": routing_buffers["output1"],
            "slot_to_expanded": routing_buffers["output2"],
            "expanded_to_slot": routing_buffers["output3"],
        }
    return {
        "token_final_scales": token_final_scales,
        "token_selected_experts": token_selected_experts,
        "tile_idx_to_expert_idx": routing_buffers["output0"],
        "tile_idx_to_mn_limit": routing_buffers["output1"],
        "expanded_idx_to_permuted_idx": (
            expanded_to_permuted if emit_expanded_to_permuted else None
        ),
        "permuted_idx_to_expanded_idx": routing_buffers["output2"],
        "num_non_exiting_tiles": routing_buffers["output3"],
    }


def moe_sort(
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    local_expert_offset: int = 0,
    num_local_experts: Optional[int] = None,
    tile_tokens_dim: int = 128,
    enable_pdl: bool = False,
    # CUDA graph support: pre-allocated output buffers
    out_tile_idx_to_expert_idx: Optional[torch.Tensor] = None,
    out_tile_idx_to_mn_limit: Optional[torch.Tensor] = None,
    out_expanded_idx_to_permuted_idx: Optional[torch.Tensor] = None,
    out_permuted_idx_to_expanded_idx: Optional[torch.Tensor] = None,
    out_total_num_padded_tokens: Optional[torch.Tensor] = None,
    out_num_non_exiting_tiles: Optional[torch.Tensor] = None,
    global_counts: Optional[torch.Tensor] = None,
    global_offsets: Optional[torch.Tensor] = None,
    grid_sync: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,  # tile_idx_to_expert_idx
    torch.Tensor,  # tile_idx_to_mn_limit
    torch.Tensor,  # expanded_idx_to_permuted_idx
    torch.Tensor,  # permuted_idx_to_expanded_idx
    torch.Tensor,  # total_num_padded_tokens [1], int32 (device tensor for CUDA graph compatibility)
    torch.Tensor,  # num_non_exiting_tiles
]:
    """
    Sort tokens by expert assignment and generate mapping tensors.

    This function performs token sorting and index mapping computation required
    for grouped GEMM operations in MoE. It uses the same algorithm as TRT-LLM's
    moe_sort with DeepSeekV3 routing method.

    Note: This function does NOT physically reorder data - use moe_permute() for that.

    CUDA Graph Compatibility:
        For CUDA graph capture, pre-allocate output buffers BEFORE capture using
        allocate_moe_sort_buffers() and pass them via the out_* parameters. This
        ensures the same memory addresses are used during capture and replay.

        Example:
            >>> buffers = allocate_moe_sort_buffers(num_tokens, num_experts, top_k, ...)
            >>> # Warmup before capture
            >>> for _ in range(3):
            ...     moe_sort(..., **buffers)
            >>> # Capture
            >>> with torch.cuda.graph(g):
            ...     moe_sort(..., **buffers)

    Args:
        token_selected_experts: Expert assignments of shape [num_tokens, top_k].
                               Data type: torch.int32.
        token_final_scales: Routing weights of shape [num_tokens, top_k].
                           Data type: torch.float32 or torch.bfloat16.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        local_expert_offset: Expert offset for expert parallelism. Default: 0.
        num_local_experts: Number of local experts. Default: num_experts.
        tile_tokens_dim: Tile size for scheduling. Default: 128.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
        out_tile_idx_to_expert_idx: Pre-allocated buffer for tile_idx_to_expert_idx.
        out_tile_idx_to_mn_limit: Pre-allocated buffer for tile_idx_to_mn_limit.
        out_expanded_idx_to_permuted_idx: Pre-allocated buffer for expanded_idx_to_permuted_idx.
        out_permuted_idx_to_expanded_idx: Pre-allocated buffer for permuted_idx_to_expanded_idx.
        out_total_num_padded_tokens: Pre-allocated buffer for total_num_padded_tokens.
        out_num_non_exiting_tiles: Pre-allocated buffer for num_non_exiting_tiles.
        global_counts: Pre-allocated buffer for global_counts.
        global_offsets: Pre-allocated buffer for global_offsets.
        grid_sync: Pre-allocated buffer for grid_sync.

    Returns:
        tuple: A tuple of 6 elements:
            - tile_idx_to_expert_idx: [max_num_tiles], int32
                Mapping from tile index to local expert index (0 to num_local_experts-1).
            - tile_idx_to_mn_limit: [max_num_tiles], int32
                M/N limit for each tile (cumulative token count).
            - expanded_idx_to_permuted_idx: [num_tokens, top_k], int32
                Mapping from expanded index to permuted index.
                -1 indicates a masked/non-local expert.
            - permuted_idx_to_expanded_idx: [max_num_permuted_tokens], int32
                Mapping from permuted index to expanded index.
            - total_num_padded_tokens: [1], int32 (device tensor)
                Total number of padded tokens. Returned as tensor for CUDA graph compatibility.
            - num_non_exiting_tiles: [1], int32 (device tensor)
                Number of non-exiting (active) tiles.

    Example:
        >>> import torch
        >>> from flashinfer.cute_dsl_moe_utils import moe_sort
        >>> from flashinfer.fused_moe.utils import make_random_topk_ids
        >>>
        >>> num_tokens, num_experts, top_k = 128, 8, 2
        >>> token_selected_experts = make_random_topk_ids(num_experts, num_tokens, top_k, device="cuda")
        >>> token_final_scales = torch.randn(num_tokens, top_k, device="cuda")
        >>>
        >>> (tile_idx_to_expert_idx, tile_idx_to_mn_limit,
        ...  expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx,
        ...  total_num_padded_tokens, num_non_exiting_tiles) = moe_sort(
        ...     token_selected_experts, token_final_scales,
        ...     num_experts=num_experts, top_k=top_k)
    """
    # Validate inputs
    assert token_selected_experts.dim() == 2, "token_selected_experts must be 2D"
    assert token_final_scales.dim() == 2, "token_final_scales must be 2D"

    num_tokens = token_selected_experts.size(0)
    assert token_selected_experts.size(1) == top_k, (
        "token_selected_experts.size(1) must equal top_k"
    )
    assert token_final_scales.size(0) == num_tokens, (
        "token_final_scales.size(0) must equal num_tokens"
    )
    assert token_final_scales.size(1) == top_k, (
        "token_final_scales.size(1) must equal top_k"
    )
    if token_selected_experts.device != token_final_scales.device:
        raise ValueError("routing tensors must be on the same device")
    if token_final_scales.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("token_final_scales must have dtype float32 or bfloat16")

    if num_local_experts is None:
        num_local_experts = num_experts

    device = token_selected_experts.device

    # Calculate buffer sizes
    max_num_tiles = get_max_num_tiles(
        num_tokens, top_k, num_local_experts, tile_tokens_dim
    )
    max_num_permuted_tokens = get_max_num_permuted_tokens(
        num_tokens, top_k, num_local_experts, tile_tokens_dim
    )

    # Ensure inputs are contiguous and correct dtypes
    token_selected_experts = token_selected_experts.contiguous()
    if token_selected_experts.dtype != torch.int32:
        token_selected_experts = token_selected_experts.to(torch.int32)

    token_final_scales = token_final_scales.contiguous()

    # Use pre-allocated buffers if provided, otherwise allocate new ones
    # Pre-allocation is required for CUDA graph compatibility
    if out_tile_idx_to_expert_idx is not None:
        tile_idx_to_expert_idx = out_tile_idx_to_expert_idx
        # Zero-fill to ensure safe defaults for entries beyond num_non_exiting_tiles.
        # This prevents out-of-bounds weight accesses when Rubin kernels round up
        # the tile count to an even number for cluster synchronization.
        tile_idx_to_expert_idx.zero_()
    else:
        tile_idx_to_expert_idx = torch.zeros(
            (max_num_tiles,), dtype=torch.int32, device=device
        )

    if out_tile_idx_to_mn_limit is not None:
        tile_idx_to_mn_limit = out_tile_idx_to_mn_limit
        # Zero-fill for the same reason as tile_idx_to_expert_idx above: the Rubin
        # even-tile rounding can read one mn_limit slot the routing kernel never
        # wrote; a stale value there would corrupt row stores.
        tile_idx_to_mn_limit.zero_()
    else:
        tile_idx_to_mn_limit = torch.zeros(
            (max_num_tiles,), dtype=torch.int32, device=device
        )

    if out_expanded_idx_to_permuted_idx is not None:
        expanded_idx_to_permuted_idx = out_expanded_idx_to_permuted_idx
    else:
        expanded_idx_to_permuted_idx = torch.empty(
            (num_tokens, top_k), dtype=torch.int32, device=device
        )

    if out_permuted_idx_to_expanded_idx is not None:
        permuted_idx_to_expanded_idx = out_permuted_idx_to_expanded_idx
    else:
        permuted_idx_to_expanded_idx = torch.empty(
            (max_num_permuted_tokens,), dtype=torch.int32, device=device
        )

    if out_total_num_padded_tokens is not None:
        total_num_padded_tokens_tensor = out_total_num_padded_tokens
    else:
        total_num_padded_tokens_tensor = torch.empty(
            (1,), dtype=torch.int32, device=device
        )

    if out_num_non_exiting_tiles is not None:
        num_non_exiting_tiles = out_num_non_exiting_tiles
    else:
        num_non_exiting_tiles = torch.empty((1,), dtype=torch.int32, device=device)

    compiled, num_ctas = compile_moe_sort(
        num_tokens,
        num_experts,
        top_k,
        local_expert_offset,
        num_local_experts,
        tile_tokens_dim,
        enable_pdl,
        token_final_scales.dtype,
        device,
    )
    if global_counts is None:
        global_counts = torch.zeros(
            (2 * num_local_experts,), dtype=torch.int32, device=device
        )
    if global_offsets is None:
        global_offsets = torch.empty(
            (num_ctas * num_local_experts,),
            dtype=torch.int32,
            device=device,
        )
    if grid_sync is None:
        grid_sync = torch.zeros((4,), dtype=torch.int32, device=device)
    compiled(
        token_selected_experts,
        token_final_scales,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens_tensor,
        num_non_exiting_tiles,
        global_counts,
        global_offsets,
        grid_sync,
        cuda.CUstream(_get_cuda_stream_ptr()),
    )

    return (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens_tensor,
        num_non_exiting_tiles,
    )


@functools.lru_cache(maxsize=128)
def compile_moe_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    padded_m: Optional[int],
    tile_size: Optional[int],
    capacity: Optional[int],
    use_pdl: bool,
    emit_expanded_to_permuted: bool,
    dtype: torch.dtype,
    device: torch.device,
):
    from .blackwell.moe_routing import MoeRoutingKernel

    with torch.cuda.device(device):
        max_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        kernel = MoeRoutingKernel(
            num_tokens,
            num_experts,
            top_k,
            padded_m=padded_m,
            tile_size=tile_size,
            capacity=capacity,
            use_pdl=use_pdl,
            compact_topk=dtype != torch.float32 and num_tokens >= 64,
            emit_expanded_to_permuted=emit_expanded_to_permuted,
            max_ctas=max_ctas,
        )
        fixed_slot = capacity is not None
        if fixed_slot:
            output_shapes = (
                (num_experts * int(capacity),),
                (num_experts,),
                (num_experts * int(capacity),),
                (num_tokens * top_k,),
            )
        else:
            if padded_m is None or tile_size is None:
                raise ValueError("tiled routing requires padded_m and tile_size")
            output_shapes = (
                (padded_m // tile_size,),
                (padded_m // tile_size,),
                (padded_m,),
                (1,),
            )
        output_types = (
            (cutlass.Int64, cutlass.Int32, cutlass.Int32, cutlass.Int32)
            if fixed_slot
            else (cutlass.Int32,) * 4
        )

        def fake(dtype, shape, align=16):
            kwargs = {"assumed_align": align}
            if len(shape) == 2:
                kwargs["stride_order"] = (1, 0)
            return make_fake_compact_tensor(dtype, shape, **kwargs)

        dynamic_tokens = cute.sym_int()
        global_count_size = (
            2 * num_experts
            if not fixed_slot and num_tokens > 2048
            else kernel.num_ctas * num_experts
        )
        return cute.compile(
            kernel,
            fake(_cutlass_type(dtype), (dynamic_tokens, num_experts)),
            fake(cutlass.Float32, (dynamic_tokens, top_k)),
            fake(cutlass.Int32, (dynamic_tokens, top_k)),
            *(
                fake(output_type, shape, 4 if shape == (1,) else 16)
                for output_type, shape in zip(output_types, output_shapes, strict=True)
            ),
            fake(
                cutlass.Int32,
                (dynamic_tokens, top_k) if emit_expanded_to_permuted else (1,),
                4 if not emit_expanded_to_permuted else 16,
            ),
            fake(cutlass.Int32, (global_count_size,)),
            fake(cutlass.Int32, (kernel.num_ctas * num_experts,)),
            fake(cutlass.Int32, (num_tokens * top_k,)),
            fake(cutlass.Int32, (2,)),
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )


@functools.lru_cache(maxsize=128)
def compile_moe_sort(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    num_local_experts: int,
    tile_tokens_dim: int,
    enable_pdl: bool,
    scale_dtype: torch.dtype,
    device: torch.device,
):
    from .blackwell.moe_sort import MoeSortKernel

    with torch.cuda.device(device):
        kernel = MoeSortKernel(
            num_tokens,
            num_experts,
            top_k,
            local_expert_offset,
            num_local_experts,
            tile_tokens_dim,
            use_pdl=enable_pdl,
        )
        max_num_tiles = get_max_num_tiles(
            num_tokens, top_k, num_local_experts, tile_tokens_dim
        )
        max_num_permuted_tokens = max_num_tiles * tile_tokens_dim

        def fake_i32(shape, align=16):
            kwargs = {"assumed_align": align}
            if len(shape) == 2:
                kwargs["stride_order"] = (1, 0)
            return make_fake_compact_tensor(cutlass.Int32, shape, **kwargs)

        scale_type = (
            cutlass.Float32 if scale_dtype == torch.float32 else cutlass.BFloat16
        )
        dynamic_tokens = cute.sym_int()
        return (
            cute.compile(
                kernel,
                fake_i32((dynamic_tokens, top_k)),
                make_fake_compact_tensor(
                    scale_type,
                    (dynamic_tokens, top_k),
                    assumed_align=16,
                    stride_order=(1, 0),
                ),
                fake_i32((max_num_tiles,)),
                fake_i32((max_num_tiles,)),
                fake_i32((num_tokens, top_k)),
                fake_i32((max_num_permuted_tokens,)),
                fake_i32((1,), align=4),
                fake_i32((1,), align=4),
                fake_i32((2 * num_local_experts,)),
                fake_i32((kernel.num_ctas * num_local_experts,)),
                fake_i32((4,)),
                make_fake_stream(),
                options="--enable-tvm-ffi",
            ),
            kernel.num_ctas,
        )


def prepare_moe_routing(
    router_logits: Optional[torch.Tensor],
    token_final_scales: torch.Tensor,
    token_selected_experts: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int,
    tile_tokens_dim: int,
    use_fused_finalize: bool,
    enable_pdl: bool = False,
    routing_cache: Optional[dict[tuple, Dict[str, torch.Tensor]]] = None,
    moe_sort_buffers: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """Compute top-k routing and the grouped-GEMM row mappings."""
    if router_logits is None:
        (
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            _,
            num_non_exiting_tiles,
        ) = moe_sort(
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=local_expert_offset,
            num_local_experts=num_local_experts,
            tile_tokens_dim=tile_tokens_dim,
            enable_pdl=enable_pdl,
            **(moe_sort_buffers or {}),
        )
        return {
            "token_final_scales": token_final_scales,
            "token_selected_experts": token_selected_experts,
            "tile_idx_to_expert_idx": tile_idx_to_expert_idx,
            "tile_idx_to_mn_limit": tile_idx_to_mn_limit,
            "expanded_idx_to_permuted_idx": expanded_idx_to_permuted_idx,
            "permuted_idx_to_expanded_idx": permuted_idx_to_expanded_idx,
            "num_non_exiting_tiles": num_non_exiting_tiles,
        }
    if num_local_experts != num_experts or local_expert_offset != 0:
        raise ValueError("router logits require all experts to be local")
    token_capacity = get_token_capacity(router_logits.shape[0])
    padded_m = get_max_num_permuted_tokens(
        token_capacity,
        top_k,
        num_local_experts,
        tile_tokens_dim,
    )
    key = (
        router_logits.device,
        router_logits.dtype,
        token_capacity,
        num_experts,
        top_k,
        padded_m,
        tile_tokens_dim,
        not use_fused_finalize,
    )
    routing_cache = routing_cache if routing_cache is not None else {}
    routing_buffers = routing_cache.get(key)
    if routing_buffers is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MoE routing token bucket is not initialized for CUDA graph "
                "capture; warm up this token shape before capture"
            )
        routing_buffers = allocate_moe_routing_buffers(
            token_capacity,
            num_experts,
            top_k,
            router_logits.device,
            padded_m=padded_m,
            tile_size=tile_tokens_dim,
            emit_expanded_to_permuted=not use_fused_finalize,
        )
        routing_cache[key] = routing_buffers
    return moe_routing(
        router_logits,
        top_k,
        padded_m=padded_m,
        tile_size=tile_tokens_dim,
        use_pdl=enable_pdl,
        emit_expanded_to_permuted=not use_fused_finalize,
        token_capacity=token_capacity,
        routing_buffers=routing_buffers,
    )


# ============================== Activation Functions ==============================


def moe_activation(
    input: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    activation_type: MoeActivationType,
    max_num_permuted_tokens: int,
    tile_size: int,
    enable_pdl: bool = False,
) -> None:
    """
    Apply activation function to MoE intermediate outputs.

    This is a generic activation function that supports multiple activation types.
    For convenience, use the specific wrappers like moe_swiglu(), moe_gelu(), etc.

    Args:
        input: Input tensor. For GLU activations (Swiglu, Geglu), shape is
               [num_permuted_tokens, 2 * interm_size] where first half is linear
               projection and second half is gate. For non-GLU activations,
               shape is [num_permuted_tokens, interm_size].
               Supported dtypes: float16, bfloat16.
        output: Output tensor of shape [num_permuted_tokens, interm_size].
        tile_idx_to_mn_limit: Valid token count per tile from moe_sort.
                             Shape: [num_tiles].
        num_non_exiting_tiles: Number of valid tiles (scalar on device).
        activation_type: Type of activation to apply. See MoeActivationType.
        max_num_permuted_tokens: Maximum number of permuted tokens.
        tile_size: Tile size for scheduling.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
    """
    module = _get_moe_utils_module()
    dtype_suffix = _get_dtype_suffix(input.dtype)

    interm_size = output.shape[-1]

    func_name = f"flashinfer_moe_activation_{dtype_suffix}"
    func = module[func_name]

    func(
        input.data_ptr(),
        output.data_ptr(),
        tile_idx_to_mn_limit.data_ptr(),
        num_non_exiting_tiles.data_ptr(),
        int(activation_type),
        max_num_permuted_tokens,
        interm_size,
        tile_size,
        enable_pdl,
    )


def moe_swiglu(
    input: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    max_num_permuted_tokens: int,
    tile_size: int,
    enable_pdl: bool = False,
) -> None:
    """
    Apply SwiGLU activation for MoE intermediate outputs.

    SwiGLU(x, gate) = SiLU(gate) * x = gate * sigmoid(gate) * x

    Args:
        input: Input tensor of shape [num_permuted_tokens, 2 * interm_size].
               First half is the linear projection, second half is the gate.
               Supported dtypes: float16, bfloat16.
        output: Output tensor of shape [num_permuted_tokens, interm_size].
        tile_idx_to_mn_limit: Valid token count per tile from moe_sort.
                             Shape: [num_tiles].
        num_non_exiting_tiles: Number of valid tiles (scalar on device).
        max_num_permuted_tokens: Maximum number of permuted tokens.
        tile_size: Tile size for scheduling.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
    """
    moe_activation(
        input=input,
        output=output,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        activation_type=MoeActivationType.Swiglu,
        max_num_permuted_tokens=max_num_permuted_tokens,
        tile_size=tile_size,
        enable_pdl=enable_pdl,
    )


def moe_geglu(
    input: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    max_num_permuted_tokens: int,
    tile_size: int,
    enable_pdl: bool = False,
) -> None:
    """
    Apply GeGLU activation for MoE intermediate outputs.

    GeGLU(x, gate) = GELU(gate) * x

    Args:
        input: Input tensor of shape [num_permuted_tokens, 2 * interm_size].
               First half is the linear projection, second half is the gate.
               Supported dtypes: float16, bfloat16.
        output: Output tensor of shape [num_permuted_tokens, interm_size].
        tile_idx_to_mn_limit: Valid token count per tile from moe_sort.
                             Shape: [num_tiles].
        num_non_exiting_tiles: Number of valid tiles (scalar on device).
        max_num_permuted_tokens: Maximum number of permuted tokens.
        tile_size: Tile size for scheduling.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
    """
    moe_activation(
        input=input,
        output=output,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        activation_type=MoeActivationType.Geglu,
        max_num_permuted_tokens=max_num_permuted_tokens,
        tile_size=tile_size,
        enable_pdl=enable_pdl,
    )


def moe_gelu(
    input: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    max_num_permuted_tokens: int,
    tile_size: int,
    enable_pdl: bool = False,
) -> None:
    """
    Apply GELU activation for MoE intermediate outputs.

    GELU(x) = x * Phi(x) where Phi is the CDF of standard normal distribution.

    Args:
        input: Input tensor of shape [num_permuted_tokens, interm_size].
               Supported dtypes: float16, bfloat16.
        output: Output tensor of shape [num_permuted_tokens, interm_size].
        tile_idx_to_mn_limit: Valid token count per tile from moe_sort.
                             Shape: [num_tiles].
        num_non_exiting_tiles: Number of valid tiles (scalar on device).
        max_num_permuted_tokens: Maximum number of permuted tokens.
        tile_size: Tile size for scheduling.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
    """
    moe_activation(
        input=input,
        output=output,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        activation_type=MoeActivationType.Gelu,
        max_num_permuted_tokens=max_num_permuted_tokens,
        tile_size=tile_size,
        enable_pdl=enable_pdl,
    )


def moe_silu(
    input: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    max_num_permuted_tokens: int,
    tile_size: int,
    enable_pdl: bool = False,
) -> None:
    """
    Apply SiLU (Swish) activation for MoE intermediate outputs.

    SiLU(x) = x * sigmoid(x)

    Args:
        input: Input tensor of shape [num_permuted_tokens, interm_size].
               Supported dtypes: float16, bfloat16.
        output: Output tensor of shape [num_permuted_tokens, interm_size].
        tile_idx_to_mn_limit: Valid token count per tile from moe_sort.
                             Shape: [num_tiles].
        num_non_exiting_tiles: Number of valid tiles (scalar on device).
        max_num_permuted_tokens: Maximum number of permuted tokens.
        tile_size: Tile size for scheduling.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
    """
    moe_activation(
        input=input,
        output=output,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        activation_type=MoeActivationType.Silu,
        max_num_permuted_tokens=max_num_permuted_tokens,
        tile_size=tile_size,
        enable_pdl=enable_pdl,
    )


def moe_relu(
    input: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    max_num_permuted_tokens: int,
    tile_size: int,
    enable_pdl: bool = False,
) -> None:
    """
    Apply ReLU activation for MoE intermediate outputs.

    ReLU(x) = max(0, x)

    Args:
        input: Input tensor of shape [num_permuted_tokens, interm_size].
               Supported dtypes: float16, bfloat16.
        output: Output tensor of shape [num_permuted_tokens, interm_size].
        tile_idx_to_mn_limit: Valid token count per tile from moe_sort.
                             Shape: [num_tiles].
        num_non_exiting_tiles: Number of valid tiles (scalar on device).
        max_num_permuted_tokens: Maximum number of permuted tokens.
        tile_size: Tile size for scheduling.
        enable_pdl: Enable Programmatic Dependent Launch for better kernel overlap.
                    Default is False.
    """
    moe_activation(
        input=input,
        output=output,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        activation_type=MoeActivationType.Relu,
        max_num_permuted_tokens=max_num_permuted_tokens,
        tile_size=tile_size,
        enable_pdl=enable_pdl,
    )
