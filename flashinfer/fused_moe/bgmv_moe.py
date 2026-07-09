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
from typing import List, Optional

import torch

from ..api_logging import flashinfer_api


@functools.cache
def _get_bgmv_moe_module():
    """Lazily load the BGMV MoE CUDA extension.

    Tries in order:
    Loads via FlashInfer's JIT compilation system (TVM-FFI).
    """
    try:
        from ..jit.bgmv_moe import load_bgmv_moe_module

        return load_bgmv_moe_module()
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        raise ImportError(
            f"Failed to load BGMV MoE CUDA extension via JIT. "
            f"Ensure CUDA toolkit is available and csrc/bgmv_moe/ sources exist.\n"
            f"Error: {e}"
        ) from e


@functools.cache
def has_bgmv_moe() -> bool:
    """Return True if the BGMV MoE CUDA extension is available."""
    try:
        _get_bgmv_moe_module()
        return True
    except ImportError:
        return False


@flashinfer_api
def bgmv_moe_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    lora_stride: int,
    *,
    per_pair_input: bool = False,
) -> None:
    """
    MoE LoRA shrink operation: project input through LoRA-A matrices.

    For each (token, expert) pair, computes:
        y[slice, pair, rank] += x[token] @ lora_a[expert, lora_id, :, :]

    Args:
        y: Output tensor [num_slices, num_pairs, rank]. Accumulated in-place.
        x: Input activations [num_tokens, hidden_dim].
        w_ptr: Pointer table [num_slices, num_experts] of int64.
            Each entry points to the start of lora_a weights for (slice, expert).
            The kernel uses lora_stride to index different LoRA adapters.
        sorted_token_ids: Token indices for each pair [num_pairs].
        expert_ids: Expert indices for each pair [num_pairs].
        lora_indices: LoRA adapter ID for each token [num_tokens].
            -1 means no LoRA (pair is skipped).
        lora_stride: Stride (in elements) between consecutive LoRA adapters
            in the weight tensor. For layout [max_loras, num_experts, rank, feat],
            this is num_experts * rank * feat.
        per_pair_input: If False (default, FC1), the input row is the token, so a
            token's hidden row is reused across its k pairs (``x`` is ``[num_tokens, feat_in]``).
            If True (FC2), the input row is the pair itself, i.e. ``x`` is a per-pair
            ``[num_pairs, feat_in]`` buffer (e.g. the gathered post-activation). The
            ``lora_indices``/skip lookup still uses ``sorted_token_ids[pair]``.
    """
    mod = _get_bgmv_moe_module()
    mod.bgmv_moe_shrink(
        y,
        x,
        w_ptr,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        lora_stride,
        per_pair_input,
    )


@flashinfer_api
def bgmv_moe_expand(
    y: torch.Tensor,
    x: torch.Tensor,
    w_ptr: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_indices: torch.Tensor,
    slice_start_loc: torch.Tensor,
    output_slices: List[int],
    lora_stride: int,
    *,
    finalize: bool = True,
) -> None:
    """
    MoE LoRA expand operation: project through LoRA-B matrices.

    With ``finalize=True`` (default), for each (token, expert) pair computes the
    routing-weighted combine into a per-token row:
        y[token, col_offset:col_offset+feat] += topk_weight * (x[slice, pair, :] @ lora_b[expert, lora_id])
    (``y`` is ``[num_tokens, total_feat_out]`` and must be zero-initialized).

    With ``finalize=False`` (FC1 LoRA delta), writes a per-pair, UNWEIGHTED result with a
    plain store — no ``topk_weight``, no cross-expert combine:
        y[pair, col_offset:col_offset+feat] = (x[slice, pair, :] @ lora_b[expert, lora_id])
    (``y`` is ``[num_pairs, total_feat_out]``). Skipped pairs (lora_id < 0) early-return, so
    ``y`` MUST be zero-initialized by the caller (``torch.zeros``) to define those rows.
    ``topk_weights`` is ignored in this mode but must still be a valid ``[num_pairs]`` float32
    tensor.

    Args:
        y: Output buffer (zero-initialized). ``[num_tokens, total_feat_out]`` (finalize) or
            ``[num_pairs, total_feat_out]`` (no-finalize). Float32.
        x: Shrink output [num_slices, num_pairs, rank].
        w_ptr: Pointer table [num_slices, num_experts] of int64.
        sorted_token_ids: Token indices for each pair [num_pairs].
        expert_ids: Expert indices for each pair [num_pairs].
        topk_weights: Routing weights for each pair [num_pairs]. Float32. (Ignored when
            ``finalize=False``.)
        lora_indices: LoRA adapter ID for each token [num_tokens].
        slice_start_loc: Column offset for each slice [num_slices]. Int64.
        output_slices: Output feature dimension for each slice.
        lora_stride: Stride between LoRA adapters in weight tensor.
        finalize: Combine + weight per token (True) vs per-pair unweighted store (False).
    """
    mod = _get_bgmv_moe_module()
    mod.bgmv_moe_expand(
        y,
        x,
        w_ptr,
        sorted_token_ids,
        expert_ids,
        topk_weights,
        lora_indices,
        slice_start_loc,
        output_slices[0],
        lora_stride,
        finalize,
    )


def fill_w_ptr(
    w_ptr: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    slice_id: int,
) -> int:
    """
    Fill the weight pointer table for a given slice.

    Populates w_ptr[slice_id, 0:num_experts] with data pointers for each expert.
    Works with weight layout [max_loras, num_experts, rank, feat].

    Args:
        w_ptr: Pointer table [num_slices, num_experts] of int64.
        weights: LoRA weight tensor [max_loras, num_experts, rank, feat].
        num_experts: Number of experts.
        slice_id: Which slice to populate.

    Returns:
        lora_stride: The stride (in elements) between LoRA adapters.
    """
    # w shape: [max_loras, num_experts, rank, feat]
    base_ptr = weights.data_ptr()
    expert_stride_bytes = weights.stride(1) * weights.element_size()

    arange = torch.arange(num_experts, dtype=torch.int64, device=weights.device)
    w_ptr[slice_id, :num_experts] = arange * expert_stride_bytes + base_ptr

    # lora_stride = stride along dim 0 (in elements)
    return weights.stride(0)


@flashinfer_api
def bgmv_moe(
    x: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    lora_b_weights: List[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    output_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    High-level multi-LoRA MoE BGMV: shrink + expand in one call.

    Computes the LoRA delta for MoE:
        delta[token] = Σ_expert (topk_weight * x[token] @ lora_a[expert, lora_id] @ lora_b[expert, lora_id])

    Args:
        x: Input activations [num_tokens, hidden_dim].
        lora_a_weights: List of LoRA-A weight tensors, one per slice.
            Each has shape [max_loras, num_experts, rank, hidden_dim].
        lora_b_weights: List of LoRA-B weight tensors, one per slice.
            Each has shape [max_loras, num_experts, feat_out, rank].
        sorted_token_ids: Token indices for each pair [num_pairs].
        expert_ids: Expert indices for each pair [num_pairs].
        lora_indices: LoRA adapter ID for each token [num_tokens].
        topk_weights: Routing weights for each pair [num_pairs].
        num_experts: Number of experts.
        output_dim: Total output dimension. If None, inferred from lora_b_weights.

    Returns:
        Output tensor [num_tokens, total_feat_out] with LoRA deltas.
    """
    num_slices = len(lora_a_weights)
    num_tokens = x.size(0)
    num_pairs = sorted_token_ids.size(0)
    rank = lora_a_weights[0].size(2)
    device = x.device
    dtype = x.dtype

    # Infer output dimension
    feat_out_per_slice = [lora_b_weights[s].size(2) for s in range(num_slices)]
    total_feat_out = output_dim if output_dim is not None else sum(feat_out_per_slice)

    # Build w_ptr for shrink (lora_a)
    w_ptr_a = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride_a = 0
    for s in range(num_slices):
        lora_stride_a = fill_w_ptr(w_ptr_a, lora_a_weights[s], num_experts, s)

    # Shrink: x @ lora_a -> [num_slices, num_pairs, rank]
    shrink_out = torch.zeros(num_slices, num_pairs, rank, dtype=dtype, device=device)
    bgmv_moe_shrink(
        shrink_out,
        x,
        w_ptr_a,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        lora_stride_a,
    )

    # Build w_ptr for expand (lora_b)
    w_ptr_b = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride_b = 0
    for s in range(num_slices):
        lora_stride_b = fill_w_ptr(w_ptr_b, lora_b_weights[s], num_experts, s)

    # Slice start locations (build on CPU, transfer once to avoid per-element sync)
    slice_start_loc_cpu = torch.zeros(num_slices, dtype=torch.int64)
    loc = 0
    for s in range(num_slices):
        slice_start_loc_cpu[s] = loc
        loc += feat_out_per_slice[s]
    slice_start_loc = slice_start_loc_cpu.to(device=device)

    # Expand: shrink_out @ lora_b -> [num_tokens, total_feat_out]
    y = torch.zeros(num_tokens, total_feat_out, dtype=torch.float32, device=device)
    bgmv_moe_expand(
        y,
        shrink_out,
        w_ptr_b,
        sorted_token_ids,
        expert_ids,
        topk_weights,
        lora_indices,
        slice_start_loc,
        feat_out_per_slice,
        lora_stride_b,
    )

    return y.to(dtype)
