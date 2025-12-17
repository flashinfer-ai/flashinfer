"""
Copyright (c) 2023 by FlashInfer team.

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

import math
import random
import time
from typing import Tuple, Any, List, Optional

import os
import sys
import warnings

import numpy as np
import torch
from einops import rearrange, reduce, repeat

from flashinfer.utils import round_up


# =============================================================================
# Rotating Buffer Utilities for Cold-L2 Benchmarking
# =============================================================================


def get_l2_cache_size(device=None) -> int:
    """
    Get L2 cache size in bytes for the given CUDA device.

    Args:
        device: CUDA device (int, torch.device, or None for current device).

    Returns:
        L2 cache size in bytes.
    """
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.L2_cache_size


def _calculate_tensor_bytes(tensors: List[torch.Tensor]) -> int:
    """
    Calculate total bytes of tensors residing on GPU.
    Assumes all tensors are on the same device.

    Args:
        tensors: List of torch.Tensor objects.

    Returns:
        Total bytes occupied by GPU tensors (CPU tensors are ignored).
    """
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            total += t.numel() * t.element_size()
    return total


def _extract_gpu_tensors(obj) -> List[torch.Tensor]:
    """
    Recursively extract all GPU-resident tensors from a nested structure
    of lists, tuples, and dicts.

    Args:
        obj: Object to extract tensors from (can be tensor, list, tuple, dict, or other).

    Returns:
        Flat list of tensors on GPU found in the structure.
    """
    tensors = []
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_extract_gpu_tensors(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            tensors.extend(_extract_gpu_tensors(v))
    return tensors


def calculate_rotation_count(
    tensors: List[torch.Tensor], device=None, min_rotations: int = 2
) -> int:
    """
    Calculate the number of buffer copies needed to ensure cold L2 cache.

    The function uses conservative thresholds to account for:
    - LRU eviction being gradual (not all data evicted when capacity exceeded)
    - Cache associativity effects (some data may persist in non-conflicting sets)
    - Hardware prefetching behavior

    Returns 1 (no rotation needed) only when tensor size substantially exceeds
    L2 cache (>= 5x), ensuring cache effects are truly negligible.

    Args:
        tensors: List of tensors to consider for rotation (must be on GPU).
        device: Device for L2 cache query (None for current device).
        min_rotations: Minimum number of rotations when rotation is needed.

    Returns:
        Number of buffer copies needed (1 means no rotation needed).
    """
    l2_size = get_l2_cache_size(device)
    total_bytes = _calculate_tensor_bytes(tensors)

    if total_bytes == 0:
        return 1  # No tensors to rotate

    # Use aggressive threshold: only skip rotation if tensors far exceed L2 (5x)
    # This ensures cache effects are truly negligible even with prefetching
    safe_cache_threshold = l2_size * 5
    if total_bytes >= safe_cache_threshold:
        return 1  # Tensors far exceed L2, no rotation needed

    # Conservative formula: ensure between any two uses of the same buffer,
    # we've accessed enough data to fully flush L2 with margin
    # Using safe_cache_threshold ensures we account for all cache effects
    num_rotations = math.ceil(safe_cache_threshold / total_bytes) + 1

    return max(min_rotations, num_rotations)


def _clone_structure(obj):
    """
    Deep clone a nested structure, cloning GPU tensors with detach().clone()
    while preserving scalars, booleans, and other non-tensor values.

    For non-contiguous tensors (e.g., created with as_strided), this function
    preserves the stride pattern using torch.empty_strided() + copy_(). This is
    important for backends like cuDNN that expect specific memory layouts.

    Args:
        obj: Object to clone (tensor, list, tuple, dict, or other).

    Returns:
        Cloned structure with GPU tensors cloned, other values preserved.
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            if obj.is_contiguous():
                return obj.detach().clone()
            else:
                # Preserve stride pattern for non-contiguous tensors
                # (e.g., as_strided views used by cuDNN paged attention)
                result = torch.empty_strided(
                    obj.size(),
                    obj.stride(),
                    dtype=obj.dtype,
                    device=obj.device,
                )
                result.copy_(obj.detach())
                return result
        else:
            return obj  # CPU tensors returned as-is
    elif isinstance(obj, list):
        return [_clone_structure(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_clone_structure(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: _clone_structure(v) for k, v in obj.items()}
    else:
        # Non-tensor, non-container: return as-is (e.g., int, float, str, bool, None)
        return obj


def _create_rotated_buffer_copies(
    input_args: Tuple, input_kwargs: dict, num_rotations: int
) -> List[Tuple[Tuple, dict]]:
    """
    Create multiple copies of input_args and input_kwargs for buffer rotation.

    The first copy (index 0) uses the original args/kwargs.
    Subsequent copies clone all GPU tensors while preserving other values.

    Args:
        input_args: Positional arguments tuple.
        input_kwargs: Keyword arguments dict.
        num_rotations: Number of buffer copies to create.

    Returns:
        List of (args, kwargs) tuples, one for each rotation index.
    """
    if num_rotations <= 1:
        return [(input_args, input_kwargs)]

    copies = []
    # First copy uses original args/kwargs
    copies.append((input_args, input_kwargs))

    # Create cloned copies for remaining rotations
    for _ in range(num_rotations - 1):
        cloned_args = _clone_structure(input_args)
        cloned_kwargs = _clone_structure(input_kwargs)
        copies.append((cloned_args, cloned_kwargs))

    return copies


def _infer_device_from_tensors(input_args, input_kwargs, default="cuda"):
    """
    Infer CUDA device from GPU tensors in input_args/input_kwargs.

    Args:
        input_args: Positional arguments tuple.
        input_kwargs: Keyword arguments dict (can be None).
        default: Default device if no GPU tensors found.

    Returns:
        Device string or torch.device.
    """
    if input_kwargs is None:
        input_kwargs = {}
    gpu_tensors = _extract_gpu_tensors(input_args) + _extract_gpu_tensors(input_kwargs)
    if gpu_tensors:
        return gpu_tensors[0].device
    return default


def _ceil_to_ue8m0(x: torch.Tensor):
    """imported from DeepGEMM"""
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """imported from DeepGEMM"""
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 448.0)
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """imported from DeepGEMM"""
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (round_up(m, 128), round_up(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def quantize_fp8(x, scale_shape, tile_shape, scale_major_mode):
    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 scales.
    """
    # 1. Assertions and Initial Setup
    ndim = x.ndim
    assert ndim in [2, 3], f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(scale_shape) == len(tile_shape)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_amax = torch.tensor(fp8_info.max, device=x.device, dtype=torch.float32)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        s0, s1 = scale_shape
        t0, t1 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(x, "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(x_scale, "s0 s1 -> (s0 t0) (s1 t1)", t0=t0, t1=t1)
        else:
            # Handle column-major tiling
            x_tiled = rearrange(x, "(s1 t0) (s0 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 -> s1 s0")
            scales_repeated = repeat(
                scales_permuted, "s1 s0 -> (s1 t0) (s0 t1)", t0=t0, t1=t1
            )

    elif ndim == 3:
        s0, s1, s2 = scale_shape
        t0, t1, t2 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(
                x, "(s0 t0) (s1 t1) (s2 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(
                x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max"
            ).clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(
                x_scale, "s0 s1 s2 -> (s0 t0) (s1 t1) (s2 t2)", t0=t0, t1=t1, t2=t2
            )
        else:
            # Handle layout where the last two axes are swapped
            x_tiled = rearrange(
                x, "(s0 t0) (s2 t1) (s1 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(
                x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max"
            ).clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1")
            scales_repeated = repeat(
                scales_permuted,
                "s0 s2 s1 -> (s0 t0) (s2 t1) (s1 t2)",
                t0=t0,
                t1=t1,
                t2=t2,
            )

    # 3. Final Quantization
    # Divide the original tensor by the broadcasted scales
    x_fp32 = x / (scales_repeated + 1e-8)

    # Convert the result to the target FP8 format
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    return x_fp8, x_scale


def dequantize_fp8(x, x_scale, scale_major_mode):
    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 scales.
    """
    # 1. Assertions and Initial Setup
    ndim = x.ndim
    assert ndim in [2, 3], f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(x_scale.shape)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        if scale_major_mode == "K":
            s0, s1 = x_scale.shape
        else:
            s1, s0 = x_scale.shape
        x = rearrange(
            x.to(torch.float32), "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1
        )
        if scale_major_mode == "K":
            x_scale = rearrange(x_scale, "s0 s1 -> s0 s1 1 1")
        else:
            x_scale = rearrange(x_scale, "s0 s1 -> s1 s0 1 1")
        out = rearrange(x * x_scale, "s0 s1 t0 t1 -> (s0 t0) (s1 t1)")

    elif ndim == 3:
        if scale_major_mode == "K":
            s0, s1, s2 = x_scale.shape
        else:
            s0, s2, s1 = x_scale.shape
        x = rearrange(
            x.to(torch.float32),
            "(s0 t0) (s1 t1) (s2 t2)-> s0 s1 s2 t0 t1 t2",
            s0=s0,
            s1=s1,
            s2=s2,
        )
        if scale_major_mode == "K":
            x_scale = rearrange(x_scale, "s0 s1 s2 -> s0 s1 s2 1 1 1")
        else:
            x_scale = rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1 1 1 1")
        out = rearrange(x * x_scale, "s0 s1 s2 t0 t1 t2 -> (s0 t0) (s1 t1) (s2 t2)")
    return out


def set_seed(random_seed):
    """
    Set random seed for reproducibility during testing.

    Args:
        random_seed (int): Random seed to set.

    Returns:
        None
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


def sleep_after_kernel_run(execution_time):
    """
    Sleep after kernel run. Dynamically adjust sleep time up to 1 sec based on execution time.

    Args:
        execution_time (float): Kernel execution time in milliseconds.

    Returns:
        None
    """
    if not math.isinf(execution_time):
        sleep_time = np.min([execution_time / 200, 1.0])
    else:
        sleep_time = 0.01
    time.sleep(sleep_time)
    return


def attention_flops(
    batch_size,
    qo_seqlen,
    kv_seqlen,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
):
    """
    Calculate FLOPs for a given attention layer. Assumes all sequence lengths are the same within the batch

    Args:
        batch_size (int): Batch size.
        qo_seqlen (int): Sequence length of the query. Assumed same within the batch.
        kv_seqlen (int): Sequence length of the key and value. Assumed same within the batch.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        causal (bool): Whether to use causal masking. FLOPs is halved for causal masking.

    Returns:
        total_flops (int): Total FLOPs for the layer.
    """
    # Causal attention requires kv_len >= q_len
    if qo_seqlen > kv_seqlen:
        raise ValueError(
            "qo_seqlen must be less than or equal to kv_seqlen for causal attention"
        )

    if causal:
        bmm1_flops = (
            batch_size
            * (2 * kv_seqlen - qo_seqlen)
            * qo_seqlen
            * num_qo_heads
            * head_dim_qk
        )
        bmm2_flops = (
            batch_size
            * (2 * kv_seqlen - qo_seqlen)
            * qo_seqlen
            * num_qo_heads
            * head_dim_vo
        )
    else:
        bmm1_flops = 2 * batch_size * qo_seqlen * kv_seqlen * num_qo_heads * head_dim_qk
        bmm2_flops = 2 * batch_size * qo_seqlen * kv_seqlen * num_qo_heads * head_dim_vo
    total_flops = bmm1_flops + bmm2_flops
    return total_flops


def attention_flops_with_actual_seq_lens(
    actual_seq_lens_q,
    actual_seq_lens_kv,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
):
    """
    Calculate FLOPs for a given attention layer with actual sequence lengths where
    actual sequence lengths are provided as 1D tensors.

    Args:
        actual_seq_lens_q (torch.Tensor): Array of actual sequence lengths of the query.
        actual_seq_lens_kv (torch.Tensor): Array of actual sequence lengths of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        causal (bool): Whether to use causal masking.
        Note: Causal must be false for decode as this function assumes qo_seqlen == kv_seqlen.

    Returns:
        total_flops (int): Total FLOPs for the layer.
    """
    # Causal attention requires kv_len >= q_len
    # Otherwise right align if kv_len > q_len
    if causal and (actual_seq_lens_q > actual_seq_lens_kv).any():
        raise ValueError(
            "actual_seq_lens_q must be less than or equal to actual_seq_lens_kv for causal attention"
        )

    if causal:
        bmm1_flops = (
            torch.dot(
                2 * actual_seq_lens_kv.to(torch.float32)
                - actual_seq_lens_q.to(torch.float32),
                actual_seq_lens_q.to(torch.float32),
            )
            * num_qo_heads
            * head_dim_qk
        )
        bmm2_flops = (
            torch.dot(
                2 * actual_seq_lens_kv.to(torch.float32)
                - actual_seq_lens_q.to(torch.float32),
                actual_seq_lens_q.to(torch.float32),
            )
            * num_qo_heads
            * head_dim_vo
        )

    else:
        bmm1_flops = (
            2
            * torch.dot(
                actual_seq_lens_kv.to(torch.float32),
                actual_seq_lens_q.to(torch.float32),
            )
            * num_qo_heads
            * head_dim_qk
        )
        bmm2_flops = (
            2
            * torch.dot(
                actual_seq_lens_kv.to(torch.float32),
                actual_seq_lens_q.to(torch.float32),
            )
            * num_qo_heads
            * head_dim_vo
        )

    total_flops = bmm1_flops + bmm2_flops
    return total_flops


def attention_tflops_per_sec(
    batch_size,
    qo_seqlen,
    kv_seqlen,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
    time,
):
    """
    Calculate TFLOPS per second for a given attention layer. Assumes all sequence lengths are the same within the batch.

    Args:
        batch_size (int): Batch size.
        qo_seqlen (int): Sequence length of the query.
        kv_seqlen (int): Sequence length of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        causal (bool): Whether to use causal masking.
        time (float): Execution time in milliseconds.

    Returns:
        tflops_per_sec (float): TFLOPS per second for the layer.
    """
    f = attention_flops(
        batch_size,
        qo_seqlen,
        kv_seqlen,
        head_dim_qk,
        head_dim_vo,
        num_qo_heads,
        causal,
    )
    return f / time / 1e9 if not math.isnan(time) else 0.0


def attention_tflops_per_sec_with_actual_seq_lens(
    actual_seq_lens_q,
    actual_seq_lens_kv,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    causal,
    ms,
):
    """
    Calculate TFLOPS per second for a given attention layer with actual sequence lengths.
    Does not assume all sequence lengths are the same within the batch.

    Args:
        actual_seq_lens_q (torch.Tensor): Array of actual sequence lengths of the query.
        actual_seq_lens_kv (torch.Tensor): Array of actual sequence lengths of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        causal (bool): Whether to use causal masking.
        ms (float): Execution time in milliseconds.

    Returns:
        tflops_per_sec (float): TFLOPS per second for the layer.
    """
    f = attention_flops_with_actual_seq_lens(
        actual_seq_lens_q,
        actual_seq_lens_kv,
        head_dim_qk,
        head_dim_vo,
        num_qo_heads,
        causal,
    )
    return f.item() / ms / 1e9 if not math.isnan(ms) else 0.0


def attention_tb_per_sec(
    batch_size,
    qo_seqlen,
    kv_seqlen,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    num_kv_heads,
    time,
    q_dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    o_dtype=torch.bfloat16,
):
    """
    Calculate TB per second perf achieved for a given attention layer. Assumes all sequence lengths are the same within the batch.

    Args:
        batch_size (int): Batch size.
        qo_seqlen (int): Sequence length of the query.
        kv_seqlen (int): Sequence length of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        num_kv_heads (int): Number of key and value heads.
        time (float): Execution time in milliseconds.
        q_dtype (torch.dtype): Data type of the query.
        kv_dtype (torch.dtype): Data type of the key and value.
        o_dtype (torch.dtype): Data type of the output.

    Returns:
        tb_per_sec (float): TB per second for the layer.
    """
    q_bytes = batch_size * qo_seqlen * num_qo_heads * head_dim_qk * q_dtype.itemsize
    k_bytes = batch_size * kv_seqlen * num_kv_heads * head_dim_qk * kv_dtype.itemsize
    v_bytes = batch_size * kv_seqlen * num_kv_heads * head_dim_vo * kv_dtype.itemsize
    o_bytes = batch_size * qo_seqlen * num_qo_heads * head_dim_vo * o_dtype.itemsize
    total_bytes = q_bytes + k_bytes + v_bytes + o_bytes

    time_in_sec = time / 1e3
    bytes_in_tb = total_bytes / 1e12  # TB not TiB
    return bytes_in_tb / time_in_sec if not math.isnan(time) else 0.0


def attention_tb_per_sec_with_actual_seq_lens(
    actual_seq_lens_q,
    actual_seq_lens_kv,
    head_dim_qk,
    head_dim_vo,
    num_qo_heads,
    num_kv_heads,
    time,
    q_dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    o_dtype=torch.bfloat16,
):
    """
    Calculate TB per second perf achieved for a given attention layer with actual sequence lengths.
    Does not assume all sequence lengths are the same within the batch.

    Args:
        actual_seq_lens_q (torch.Tensor): Array of actual sequence lengths of the query.
        actual_seq_lens_kv (torch.Tensor): Array of actual sequence lengths of the key and value.
        head_dim_qk (int): Head dimension of the query and key.
        head_dim_vo (int): Head dimension of the value.
        num_qo_heads (int): Number of query heads.
        num_kv_heads (int): Number of key and value heads.
        time (float): Execution time in milliseconds.
        q_dtype (torch.dtype): Data type of the query.
        kv_dtype (torch.dtype): Data type of the key and value.
        o_dtype (torch.dtype): Data type of the output.

    Returns:
        tb_per_sec (float): TB per second for the layer.
    """
    q_bytes = (
        torch.sum(actual_seq_lens_q) * num_qo_heads * head_dim_qk * q_dtype.itemsize
    )
    k_bytes = (
        torch.sum(actual_seq_lens_kv) * num_kv_heads * head_dim_qk * kv_dtype.itemsize
    )
    v_bytes = (
        torch.sum(actual_seq_lens_kv) * num_kv_heads * head_dim_vo * kv_dtype.itemsize
    )
    o_bytes = (
        torch.sum(actual_seq_lens_q) * num_qo_heads * head_dim_vo * o_dtype.itemsize
    )

    total_bytes = (q_bytes + k_bytes + v_bytes + o_bytes).item()

    time_in_sec = time / 1e3
    bytes_in_tb = total_bytes / 1e12  # TB not TiB
    return bytes_in_tb / time_in_sec if not math.isnan(time) else 0.0


def bench_gpu_time_with_cuda_event(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
):
    """
    Benchmark kernel execution time using CUDA events (no CUDA graphs).

    This is the simplest benchmarking method. Best suited for kernels where launch overhead
    is negligible compared to execution time.

    The function performs:
    1. A quick estimation phase (5 iterations) to determine iteration counts
    2. Dry-run warmup iterations (not measured)
    3. Measured iterations with per-iteration timing via CUDA events

    Iteration counts can be specified directly or derived from target durations:
    - If dry_run_iters/repeat_iters are provided, those counts are used directly.
    - Otherwise, counts are computed from dry_run_time_ms/repeat_time_ms.

    Args:
        fn (Callable): The kernel function to benchmark.
        dry_run_iters (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        repeat_iters (int, optional): Number of measured iterations.
            If None, computed from repeat_time_ms.
        dry_run_time_ms (int): Target warmup duration in ms (default: 25).
        repeat_time_ms (int): Target measurement duration in ms (default: 100).
        sleep_after_run (bool): If True, sleep briefly after each iteration to
            reduce thermal throttling (default: False).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, flush L2 cache before each iteration to
            ensure cold-cache performance measurements (default: True).

    Returns:
        List[float]: Per-iteration execution times in milliseconds.

    Example:
        Basic usage:

        >>> def my_kernel(a, b):
        ...     return torch.matmul(a, b.T)
        >>> q = torch.randn(1024, 128, device="cuda")
        >>> k = torch.randn(1024, 128, device="cuda")
        >>> times = bench_gpu_time_with_cuda_event(
        ...     fn=my_kernel,
        ...     input_args=(q, k),
        ... )
        >>> print(f"Median time: {np.median(times):.3f} ms")

    Note:
        This method does NOT use CUDA graphs, so each iteration incurs kernel
        launch overhead. For microbenchmarking where launch latency matters,
        consider using ``bench_gpu_time_with_cudagraph`` instead.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device`` parameters
        are deprecated. Use ``cold_l2_cache`` instead.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        _do_l2_flush = l2_flush if l2_flush is not None else True
        _l2_flush_size_mb = l2_flush_size_mb if l2_flush_size_mb is not None else 256
        _l2_flush_device = l2_flush_device if l2_flush_device is not None else "cuda"
    else:
        _do_l2_flush = cold_l2_cache
        # Dynamically determine L2 flush size and device
        _l2_flush_device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")
        l2_size = get_l2_cache_size(_l2_flush_device)
        # Use 2x L2 size to ensure complete flush
        _l2_flush_size_mb = (l2_size * 2) // (1024 * 1024)

    # Check if args are provided (determines how we call fn)
    has_args = bool(input_args) or bool(input_kwargs)

    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    buffer = None
    if _do_l2_flush:
        l2_flush_size = int(_l2_flush_size_mb) * 1024 * 1024
        buffer = torch.empty(l2_flush_size, device=_l2_flush_device, dtype=torch.int8)

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    torch.cuda.synchronize()
    call_fn()  # Call once to exclude initial overhead
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(measurement_iters):
        if _do_l2_flush:
            buffer.zero_()
        call_fn()
    end_event.record()
    torch.cuda.synchronize()
    estimated_kernel_execution_time = (
        start_event.elapsed_time(end_event) / measurement_iters
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry runs
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        if _do_l2_flush:
            buffer.zero_()
        call_fn()
    torch.cuda.synchronize()

    # Actual run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.cuda.synchronize()
    for iter_idx in range(repeat_iters):
        if _do_l2_flush:
            buffer.zero_()
        start_events[iter_idx].record()
        call_fn()
        end_events[iter_idx].record()

        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)

    # Synchronize once outside of the loop to avoid synchronization overhead
    torch.cuda.synchronize()
    measured_times = []
    for iter_idx in range(repeat_iters):
        measured_times.append(start_events[iter_idx].elapsed_time(end_events[iter_idx]))
    return measured_times


def bench_gpu_time_with_cupti(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    use_cuda_graph: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
):
    """
    Benchmark GPU time using CUPTI activity tracing for precise kernel timing.

    CUPTI (CUDA Profiling Tools Interface) provides hardware-level profiling that
    measures actual GPU kernel execution time, excluding CPU-side launch overhead.
    This gives the most accurate kernel performance measurements.

    Cold L2 cache is achieved via L2 flush between iterations. CUPTI measures
    per-iteration, so L2 flush works correctly regardless of ``use_cuda_graph``.

    Behavior:
    - Uses CUPTI (requires version >= 13, i.e., CUDA 13+) to trace kernel activities
      and compute per-iteration GPU time from recorded start/end timestamps.
    - Optionally captures operations in a CUDA graph (use_cuda_graph=True) for
      reduced launch overhead during measurement.
    - If CUPTI is unavailable, falls back to:
      - ``bench_gpu_time_with_cudagraph`` if use_cuda_graph=True (uses rotating buffers
        for cold L2)
      - ``bench_gpu_time_with_cuda_event`` otherwise (uses L2 flush for cold L2)

    Args:
        fn (Callable): The kernel function to benchmark.
        dry_run_iters (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        repeat_iters (int, optional): Number of measured iterations.
            If None, computed from repeat_time_ms.
        dry_run_time_ms (int): Target warmup duration in ms (default: 25).
        repeat_time_ms (int): Target measurement duration in ms (default: 100).
        sleep_after_run (bool): If True, sleep briefly after each iteration (default: False).
        use_cuda_graph (bool): If True, capture and replay a CUDA graph (default: False).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, flush L2 cache before each iteration to
            ensure cold-cache performance measurements (default: True).

    Returns:
        List[float]: Per-iteration GPU kernel execution times in milliseconds.

    Example:
        Basic CUPTI benchmarking (requires cupti-python >= 13):

        >>> def my_kernel(a, b):
        ...     return torch.matmul(a, b.T)
        >>> q = torch.randn(1024, 128, device="cuda")
        >>> k = torch.randn(1024, 128, device="cuda")
        >>> times = bench_gpu_time_with_cupti(
        ...     fn=my_kernel,
        ...     input_args=(q, k),
        ... )
        >>> print(f"Median GPU time: {np.median(times):.3f} ms")

    Note:
        Requires ``cupti-python`` package version >= 13.0.0:
        ``pip install -U cupti-python``

        If CUPTI is not available, a warning is issued and the function
        automatically falls back to CUDA event or CUDA graph timing.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device`` parameters
        are deprecated. Use ``cold_l2_cache`` instead.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        _do_l2_flush = l2_flush if l2_flush is not None else True
        _l2_flush_size_mb = l2_flush_size_mb if l2_flush_size_mb is not None else 256
        _l2_flush_device = l2_flush_device if l2_flush_device is not None else "cuda"
    else:
        _do_l2_flush = cold_l2_cache
        # Dynamically determine L2 flush size and device
        _l2_flush_device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")
        l2_size = get_l2_cache_size(_l2_flush_device)
        # Use 2x L2 size to ensure complete flush
        _l2_flush_size_mb = (l2_size * 2) // (1024 * 1024)

    # check if CUPTI is installed and its version is >= 13.0.0
    try:
        from cupti import cupti
        from importlib.metadata import version as importlib_metadata_version

        cupti_version = importlib_metadata_version("cupti-python")
        if int(cupti_version.split(".")[0]) < 13:
            raise Exception(
                "CUPTI needs to be >= 13.0.0. Try 'pip install -U cupti-python'."
            )
        from functools import partial
    except (ModuleNotFoundError, Exception) as e:
        if isinstance(e, ModuleNotFoundError):
            warnings.warn(
                "CUPTI is not installed. Try 'pip install -U cupti-python'. Falling back to CUDA events for benchmarking.",
                category=UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"{e} Falling back to CUDA events for benchmarking.",
                category=UserWarning,
                stacklevel=2,
            )
        # Fallback: internally decide cold-L2 strategy based on use_cuda_graph
        if use_cuda_graph:
            # CUDA graph fallback uses rotating buffers for cold L2
            return bench_gpu_time_with_cudagraph(
                fn=fn,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
                dry_run_time_ms=dry_run_time_ms,
                repeat_time_ms=repeat_time_ms,
                sleep_after_run=sleep_after_run,
                input_args=input_args,
                input_kwargs=input_kwargs,
                cold_l2_cache=cold_l2_cache,
            )
        else:
            # Non-graph fallback uses L2 flush for cold L2
            return bench_gpu_time_with_cuda_event(
                fn=fn,
                dry_run_iters=dry_run_iters,
                repeat_iters=repeat_iters,
                dry_run_time_ms=dry_run_time_ms,
                repeat_time_ms=repeat_time_ms,
                sleep_after_run=sleep_after_run,
                input_args=input_args,
                input_kwargs=input_kwargs,
                cold_l2_cache=cold_l2_cache,
            )

    # CUPTI buffer callbacks
    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024
        max_num_records = 0
        return buffer_size, max_num_records

    def set_kernel_name(activity):
        if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
            return activity.name
        elif activity.kind == cupti.ActivityKind.MEMCPY:
            return "MEMCPY"
        elif activity.kind == cupti.ActivityKind.MEMSET:
            return "MEMSET"

    def get_bytes(activity):
        if activity.kind in (cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
            return activity.bytes
        else:
            return 0

    def get_copy_kind(activity):
        if activity.kind == cupti.ActivityKind.MEMCPY:
            return activity.copy_kind
        else:
            return 0

    def get_value(activity):
        if activity.kind == cupti.ActivityKind.MEMSET:
            return activity.value
        else:
            return 0

    def collect_kernel_info(activity):
        return (
            set_kernel_name(activity),
            activity.start,
            activity.end,
            activity.correlation_id,
            get_copy_kind(activity),
            get_bytes(activity),
            get_value(activity),
            activity.kind,
        )

    def func_buffer_completed(
        launches: list[tuple[float, float, int, int, int]],
        kernels: list[tuple[str, float, float, int, int, int, int, int]],
        activities: list,
    ):
        for activity in activities:
            if activity.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                # Kernel activity
                kernels.append(collect_kernel_info(activity))
            elif activity.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                # Runtime or Driver activity
                launches.append(
                    (
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                        activity.cbid,
                        activity.kind,
                    )
                )

    # Check if args are provided (determines how we call fn)
    has_args = bool(input_args) or bool(input_kwargs)

    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    buffer = None
    if _do_l2_flush:
        l2_flush_size = int(_l2_flush_size_mb) * 1024 * 1024
        buffer = torch.empty(l2_flush_size, device=_l2_flush_device, dtype=torch.int8)

    # Prepare runner (either direct fn or CUDA graph replay)
    runner = call_fn
    g = None
    if use_cuda_graph:
        # Warmup run to avoid capturing one-time inits
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call_fn()
        torch.cuda.current_stream().wait_stream(s)

        # Capture kernel in graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call_fn()
        runner = g.replay

    ## Estimate kernel execution time by running the runner 5 times
    measurement_iters = 5
    torch.cuda.synchronize()
    call_fn()  # Call once to exclude initial overhead
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(measurement_iters):
        if _do_l2_flush:
            buffer.zero_()
        runner()
    end_event.record()
    torch.cuda.synchronize()
    estimated_kernel_execution_time = (
        start_event.elapsed_time(end_event) / measurement_iters
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry runs
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        if _do_l2_flush:
            buffer.zero_()
        runner()
    torch.cuda.synchronize()

    # CUPTI measurement
    launches: list[tuple[float, float, int, int, int]] = []
    kernels: list[tuple[str, float, float, int, int, int, int, int]] = []
    iter_timestamps = []
    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_enable(cupti.ActivityKind.DRIVER)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)
    cupti.activity_enable(cupti.ActivityKind.MEMSET)
    cupti.activity_register_callbacks(
        func_buffer_requested, partial(func_buffer_completed, launches, kernels)
    )
    for _ in range(repeat_iters):
        if _do_l2_flush:
            buffer.zero_()
        start_cpu = cupti.get_timestamp()
        runner()
        end_cpu = cupti.get_timestamp()
        torch.cuda.synchronize()
        iter_timestamps.append((start_cpu, end_cpu))
        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)
    cupti.activity_flush_all(0)
    cupti.activity_disable(cupti.ActivityKind.RUNTIME)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_disable(cupti.ActivityKind.DRIVER)
    cupti.activity_disable(cupti.ActivityKind.MEMCPY)
    cupti.activity_disable(cupti.ActivityKind.MEMSET)
    cupti.finalize()

    def generate_kernel_string(kernel):
        # No start, end, correlation_id is considered in the kernel string
        return f"{kernel[0]}_{kernel[4]}_{kernel[5]}_{kernel[6]}_{kernel[7]}"

    # Process activities
    measured_times = []
    kernel_names = None
    for idx, (start_cpu, end_cpu) in enumerate(iter_timestamps):
        # find all launches of kernels that happened within the iteration
        iter_launches = [l for l in launches if l[0] >= start_cpu and l[0] <= end_cpu]
        corr_ids = set(l[2] for l in iter_launches)
        # find all GPU kernels that happened within the iteration
        iter_kernels = [k for k in kernels if k[3] in corr_ids]
        if not iter_kernels:
            raise ValueError(f"No kernel activities recorded for iteration {idx}")
        current_kernel_names = set(generate_kernel_string(k) for k in iter_kernels)
        # check if the kernel names are consistent
        if kernel_names is None:
            kernel_names = current_kernel_names
        else:
            if kernel_names != current_kernel_names:
                raise ValueError(
                    f"Inconsistent kernel names: {kernel_names} != {current_kernel_names}"
                )
        min_start = min(k[1] for k in iter_kernels)
        max_end = max(k[2] for k in iter_kernels)
        span_ms = (max_end - min_start) / 1e6  # ns to ms
        measured_times.append(span_ms)
    return measured_times


def bench_gpu_time_with_cudagraph(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    num_iters_within_graph: int = 10,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
):
    """
    Benchmark GPU time using CUDA graphs with amortized kernel launch overhead.

    CUDA graphs capture a sequence of GPU operations and replay them with minimal
    CPU overhead. By running multiple iterations within a single graph, kernel
    launch latency is amortized, yielding measurements closer to pure GPU time.

    **Cold-L2 Benchmarking**:

    When ``cold_l2_cache=True``, the function uses **rotating buffers** to ensure
    cold L2 cache for each kernel invocation within the graph. Multiple copies of
    the GPU tensors in ``input_args``/``input_kwargs`` are created and rotated
    through during graph capture, ensuring each kernel invocation operates on
    different memory regions. The number of buffer copies is automatically
    calculated based on the device's L2 cache size.

    Args:
        fn (Callable): The kernel function to benchmark.
        dry_run_iters (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        repeat_iters (int, optional): Number of measured iterations (graph replays).
            If None, computed from repeat_time_ms.
        dry_run_time_ms (int): Target warmup duration in ms (default: 25).
        repeat_time_ms (int): Target measurement duration in ms (default: 100).
        num_iters_within_graph (int): Number of kernel calls captured in the graph
            (default: 10). Higher values better amortize launch overhead but use
            more memory when rotating buffers.
        sleep_after_run (bool): If True, sleep briefly after each iteration (default: False).
        input_args (tuple): Positional arguments to pass to fn. GPU tensors in
            this structure will be cloned when ``cold_l2_cache=True``.
        input_kwargs (dict, optional): Keyword arguments to pass to fn. GPU tensors
            in this structure will be cloned when ``cold_l2_cache=True``.
        cold_l2_cache (bool): If True, use rotating buffers to ensure cold L2 cache
            for each kernel invocation within the graph (default: True).

    Returns:
        List[float]: Per-iteration execution times in milliseconds. Each time is
        the graph replay duration divided by ``num_iters_within_graph``.

    Example:
        Cold-L2 benchmarking (default, for memory-bound kernels):

        >>> def run_attention(q, k, v, o):
        ...     flashinfer.single_prefill_with_kv_cache(q, k, v, o)
        ...
        >>> q = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        >>> k = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        >>> v = torch.randn(batch, heads, seq_len, head_dim, device="cuda")
        >>> o = torch.empty_like(q)
        >>> times = bench_gpu_time_with_cudagraph(
        ...     fn=run_attention,
        ...     input_args=(q, k, v, o),
        ... )
        >>> print(f"Cold-L2 median time: {np.median(times):.3f} ms")

    Example:
        Hot L2 benchmarking (for compute-bound kernels):

        >>> times = bench_gpu_time_with_cudagraph(
        ...     fn=lambda: torch.matmul(q, k.T),
        ...     cold_l2_cache=False,
        ... )

    Note:
        - When using ``input_args``/``input_kwargs``, the function must accept the
          tensors as arguments (not capture them from closure).
        - GPU tensors are automatically detected and cloned. Non-tensor arguments
          (scalars, booleans, etc.) are preserved across all copies.
        - Memory usage scales with the number of rotations needed to exceed L2 cache.

    See Also:
        - ``calculate_rotation_count``: Computes required buffer copies for cold-L2.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device`` parameters
        are deprecated. Use ``cold_l2_cache`` instead.
    """
    if input_kwargs is None:
        input_kwargs = {}

    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead. For CUDA graphs, cold_l2_cache uses "
            "rotating buffers (not L2 flush) to ensure cold cache.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        # For CUDA graphs, l2_flush had limited effectiveness, so we translate
        # l2_flush=True to cold_l2_cache=True (rotating buffers)
        _do_rotate = l2_flush if l2_flush is not None else True
    else:
        _do_rotate = cold_l2_cache

    # Dynamically determine device from input tensors
    _device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")

    # Check if args are provided (determines how we call fn)
    has_args = bool(input_args) or bool(input_kwargs)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Determine rotation count if rotating buffers
    num_rotations = 1
    rotated_copies = None
    if _do_rotate:
        # Extract all GPU tensors from args and kwargs
        gpu_tensors = _extract_gpu_tensors(input_args) + _extract_gpu_tensors(
            input_kwargs
        )
        if len(gpu_tensors) == 0:
            warnings.warn(
                "cold_l2_cache=True but no GPU tensors found in input_args/input_kwargs. "
                "Cold L2 benchmarking disabled.",
                category=UserWarning,
                stacklevel=2,
            )
            _do_rotate = False
        else:
            num_rotations = calculate_rotation_count(gpu_tensors, _device)
            if num_rotations > 1:
                rotated_copies = _create_rotated_buffer_copies(
                    input_args, input_kwargs, num_rotations
                )
            else:
                # No rotation needed (tensors exceed L2)
                _do_rotate = False

    # Define how to call fn
    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    def call_fn_with_rotation(buf_idx: int):
        args, kwargs = rotated_copies[buf_idx]
        fn(*args, **kwargs)

    # Warmup run
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call_fn()
    torch.cuda.current_stream().wait_stream(s)

    # Capture kernel in graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if _do_rotate and num_rotations > 1:
            # Capture with rotating buffers: use buffer[iter % num_rotations]
            for iter_idx in range(num_iters_within_graph):
                buf_idx = iter_idx % num_rotations
                call_fn_with_rotation(buf_idx)
        else:
            # Non-rotating capture (uses original args if provided)
            for _ in range(num_iters_within_graph):
                call_fn()
    torch.cuda.synchronize()

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    start_event.record()
    for _ in range(measurement_iters):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    estimated_kernel_execution_time = (
        start_event.elapsed_time(end_event) / measurement_iters
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry run
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        g.replay()
    torch.cuda.synchronize()

    # Actual run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.cuda.synchronize()
    for iter_idx in range(repeat_iters):
        start_events[iter_idx].record()
        g.replay()
        end_events[iter_idx].record()

        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)

    # Synchronize once outside of the loop to avoid synchronization overhead
    torch.cuda.synchronize()
    measured_times = []
    for iter_idx in range(repeat_iters):
        measured_times.append(
            start_events[iter_idx].elapsed_time(end_events[iter_idx])
            / num_iters_within_graph
        )
    return measured_times


def bench_gpu_time(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    enable_cupti: bool = False,
    use_cuda_graph: bool = False,
    num_iters_within_graph: int = 10,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
):
    """
    Unified GPU benchmarking interface with configurable timing backends.

    This is the recommended entry point for GPU kernel benchmarking. It provides
    a single interface that dispatches to the appropriate timing implementation
    based on the configuration flags.

    **Timing Backends** (in order of precedence):

    1. **CUPTI** (``enable_cupti=True``): Most accurate, measures pure GPU kernel
       time via hardware profiling. Requires cupti-python >= 13.
    2. **CUDA Graphs** (``use_cuda_graph=True``): Amortizes launch overhead by
       capturing and replaying multiple kernel calls. Good balance of accuracy
       and availability.
    3. **CUDA Events** (default): Simplest method, measures launch + execution.
       Available everywhere but includes CPU overhead.

    **Cold-L2 Strategy** (automatically selected based on timing backend):

    .. list-table::
       :header-rows: 1

       * - Timing Backend
         - Cold-L2 Strategy
         - How it Works
       * - CUPTI
         - L2 Flush
         - Flush L2 cache before each iter
       * - CUDA Events (no CUDA Graphs)
         - L2 Flush
         - Flush L2 cache before each iter
       * - CUDA Events + CUDA Graphs
         - Rotating Buffers
         - Clone GPU tensors in input_args/input_kwargs and rotate through them
        use_cuda_graph (bool): If True, use CUDA graph timing (default: False).
        num_iters_within_graph (int): Kernel calls per graph (CUDA graph mode only,
            default: 10).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, ensure cold L2 cache for each iteration
            (default: True). The strategy is automatically selected based on timing
            backend.

    Returns:
        List[float]: Per-iteration execution times in milliseconds.

    Example:
        Simple benchmarking with CUDA events (default):

        >>> times = bench_gpu_time(fn=lambda: my_kernel())
        >>> print(f"Median: {np.median(times):.3f} ms")

    Example:
        CUDA graph benchmarking for reduced launch overhead:

        >>> def run_kernel(x, y, out):
        ...     my_memory_bound_kernel(x, y, out)
        >>> times = bench_gpu_time(
        ...     fn=run_kernel,
        ...     input_args=(x, y, out),
        ...     use_cuda_graph=True,
        ... )

    Example:
        CUPTI benchmarking for most accurate GPU kernel time:

        >>> times = bench_gpu_time(
        ...     fn=run_kernel,
        ...     input_args=(x, y, out),
        ...     enable_cupti=True,
        ... )

    See Also:
        - ``bench_gpu_time_with_cuda_event``: Direct CUDA event timing.
        - ``bench_gpu_time_with_cudagraph``: Direct CUDA graph timing.
        - ``bench_gpu_time_with_cupti``: Direct CUPTI timing.

    .. deprecated::
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device``
        parameters are deprecated. Use ``cold_l2_cache`` instead.
    """
    # Handle deprecated parameters
    if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
        warnings.warn(
            "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
            "Use cold_l2_cache instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        # If l2_flush was explicitly set, use it as the cold_l2_cache value
        _cold_l2_cache = l2_flush if l2_flush is not None else cold_l2_cache
    else:
        _cold_l2_cache = cold_l2_cache

    if enable_cupti:
        return bench_gpu_time_with_cupti(
            fn=fn,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            sleep_after_run=sleep_after_run,
            use_cuda_graph=use_cuda_graph,
            input_args=input_args,
            input_kwargs=input_kwargs,
            cold_l2_cache=_cold_l2_cache,
        )
    if use_cuda_graph:
        return bench_gpu_time_with_cudagraph(
            fn=fn,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            dry_run_time_ms=dry_run_time_ms,
            repeat_time_ms=repeat_time_ms,
            num_iters_within_graph=num_iters_within_graph,
            sleep_after_run=sleep_after_run,
            input_args=input_args,
            input_kwargs=input_kwargs,
            cold_l2_cache=_cold_l2_cache,
        )
    return bench_gpu_time_with_cuda_event(
        fn=fn,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        sleep_after_run=sleep_after_run,
        input_args=input_args,
        input_kwargs=input_kwargs,
        cold_l2_cache=_cold_l2_cache,
    )


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# copied from DeepGEMM
def bench_kineto(
    fn,
    kernel_names,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str = None,
    flush_l2: bool = True,
    with_multiple_kernels: bool = False,
):
    # Conflict with Nsight Systems
    using_nsys = int(os.environ.get("DG_NSYS_PROFILING", 0))

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = (
        suppress_stdout_stderr
        if suppress_kineto_output and not using_nsys
        else empty_suppress
    )
    with suppress():
        schedule = (
            torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
            if not using_nsys
            else None
        )
        profiler: Any = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
            )
            if not using_nsys
            else empty_suppress()
        )
        with profiler:
            for _i in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(
                            flush_l2_size, dtype=torch.int, device="cuda"
                        ).zero_()
                    fn()

                if not using_nsys:
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # Parse the profiling table
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    if not with_multiple_kernels:
        for name in kernel_names:
            assert sum([name in line for line in prof_lines]) == 1, (
                f"Errors of the kernel {name} in the profiling table"
            )

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {"ms": 1e3, "us": 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0.0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += (
                            float(time_str.replace(unit, "")) / scale * int(num_str)
                        )
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total
