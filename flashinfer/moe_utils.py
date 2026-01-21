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
from enum import IntEnum
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit.moe_utils import gen_moe_utils_module


# ============================ Helper Functions ============================


def get_max_num_tiles(
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    tile_size: int,
) -> int:
    """
    Calculate the maximum number of tiles for grouped GEMM.

    This follows the same logic as TRT-LLM's GroupedGemmInputsHelper.get_max_num_tiles().

    Args:
        num_tokens: Number of input tokens.
        top_k: Number of experts per token.
        num_local_experts: Number of local experts (for expert parallelism).
        tile_size: Tile size for scheduling.

    Returns:
        Maximum number of tiles.
    """
    num_expanded_tokens = num_tokens * top_k

    if num_expanded_tokens <= num_local_experts:
        return num_expanded_tokens

    # First, distribute one token to each expert
    num_remaining_tokens = num_expanded_tokens - num_local_experts
    max_num_tiles = num_local_experts

    # Greedily fill remaining tokens into tiles
    max_num_tiles += (num_remaining_tokens + tile_size - 1) // tile_size

    return max_num_tiles


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


@flashinfer_api
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
    )


@flashinfer_api
def moe_unpermute(
    permuted_input: torch.Tensor,
    output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    topk_scales: torch.Tensor,
    num_tokens: int,
    top_k: int,
    enable_pdl: bool = False,
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
        enable_pdl,
    )


@flashinfer_api
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


# ============================ moe_sort ============================


@flashinfer_api
def moe_sort(
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    local_expert_offset: int = 0,
    num_local_experts: Optional[int] = None,
    tile_tokens_dim: int = 128,
    enable_pdl: bool = False,
) -> Tuple[
    torch.Tensor,  # tile_idx_to_expert_idx
    torch.Tensor,  # tile_idx_to_mn_limit
    torch.Tensor,  # expanded_idx_to_permuted_idx
    torch.Tensor,  # permuted_idx_to_expanded_idx
    int,  # total_num_padded_tokens
    torch.Tensor,  # num_non_exiting_tiles
]:
    """
    Sort tokens by expert assignment and generate mapping tensors.

    This function performs token sorting and index mapping computation required
    for grouped GEMM operations in MoE. It uses the same algorithm as TRT-LLM's
    moe_sort with DeepSeekV3 routing method.

    Note: This function does NOT physically reorder data - use moe_permute() for that.

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
            - total_num_padded_tokens: int
                Total number of padded tokens.
            - num_non_exiting_tiles: [1], int32 (device tensor)
                Number of non-exiting (active) tiles.

    Example:
        >>> import torch
        >>> from flashinfer import moe_sort
        >>>
        >>> num_tokens, num_experts, top_k = 128, 8, 2
        >>> token_selected_experts = torch.randint(0, num_experts, (num_tokens, top_k),
        ...                                        dtype=torch.int32, device="cuda")
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

    # Allocate output tensors
    tile_idx_to_expert_idx = torch.empty(
        (max_num_tiles,), dtype=torch.int32, device=device
    )
    tile_idx_to_mn_limit = torch.empty(
        (max_num_tiles,), dtype=torch.int32, device=device
    )
    expanded_idx_to_permuted_idx = torch.full(
        (num_tokens, top_k), -1, dtype=torch.int32, device=device
    )
    permuted_idx_to_expanded_idx = torch.zeros(
        (max_num_permuted_tokens,), dtype=torch.int32, device=device
    )
    total_num_padded_tokens_tensor = torch.zeros((1,), dtype=torch.int32, device=device)
    num_non_exiting_tiles = torch.zeros((1,), dtype=torch.int32, device=device)

    # Allocate expert counts buffer for large token counts (>1024)
    # Required size: 2 * num_experts
    if num_tokens > 1024:
        expert_counts = torch.zeros(
            (2 * num_experts,), dtype=torch.int32, device=device
        )
        expert_counts_ptr = expert_counts.data_ptr()
    else:
        expert_counts_ptr = 0  # Will be set to nullptr in kernel

    # Get the JIT module and call the kernel
    module = _get_moe_utils_module()
    func = module["flashinfer_moe_sort"]

    func(
        # Inputs
        token_selected_experts.data_ptr(),
        token_final_scales.data_ptr(),
        num_tokens,
        num_experts,
        top_k,
        local_expert_offset,
        num_local_experts,
        tile_tokens_dim,
        enable_pdl,
        # Outputs
        tile_idx_to_expert_idx.data_ptr(),
        tile_idx_to_mn_limit.data_ptr(),
        expanded_idx_to_permuted_idx.data_ptr(),
        permuted_idx_to_expanded_idx.data_ptr(),
        total_num_padded_tokens_tensor.data_ptr(),
        num_non_exiting_tiles.data_ptr(),
        # Optional buffer
        expert_counts_ptr,
    )

    # Get total_num_padded_tokens as Python int
    total_num_padded_tokens = total_num_padded_tokens_tensor.item()

    return (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
    )


# ============================== Activation Functions ==============================


@flashinfer_api
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


@flashinfer_api
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


@flashinfer_api
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


@flashinfer_api
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


@flashinfer_api
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


@flashinfer_api
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


__all__ = [
    # Helper functions
    "get_max_num_tiles",
    "get_max_num_permuted_tokens",
    # Sorting/Routing
    "moe_sort",
    # Permute/Unpermute
    "moe_permute",
    "moe_unpermute",
    "moe_output_memset",
    # Activation functions
    "MoeActivationType",
    "moe_activation",
    "moe_swiglu",
    "moe_geglu",
    "moe_gelu",
    "moe_silu",
    "moe_relu",
]
