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
from typing import Tuple, Any

import os
import sys

import numpy as np
import torch
from einops import rearrange, reduce, repeat

from flashinfer.utils import round_up


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
    time,
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
        time (float): Execution time in milliseconds.

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
    return f.item() / time / 1e9 if not math.isnan(time) else 0.0


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


def bench_gpu_time(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: bool = True,
    l2_flush_size_mb: int = 256,
    l2_flush_device: str = "cuda",
    sleep_after_run: bool = False,
):
    """
    Benchmark kernel execution time without using CUDA graphs.
    Measures kernel launch latency + actual kernel execution time for fn().
    Can flush L2 cache and sleep after the run.

    Number of dry run and actual run iterations can be set by iteration count or time:
    - If dry_run_iters and repeat_iters are provided, provided iteration count will be used.
    - If dry_run_iters and repeat_iters are not provided, dry_run_time_ms and repeat_time_ms will be used.

    Returns an array of measured times so that the caller can compute statistics.

    Args:
        fn: Function to benchmark.
        dry_run_iters: Number of dry runs during which times does not count. If not provided, dry_run_time_ms will be used.
        repeat_iters: Number of iterations. If not provided, repeat_time_ms will be used.
        dry_run_time_ms: Time to run the dry run in milliseconds.
        repeat_time_ms: Time to run the repeat in milliseconds.
        l2_flush: Whether to flush L2 cache.
        l2_flush_size_mb: Size of the L2 cache to flush.
        l2_flush_device: Device that needs to flush L2 cache.
        sleep_after_run: Whether to sleep after the run. Sleep time is dynamically set.

    Returns:
        measured_times: List of measured times.
    """

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    if l2_flush:
        l2_flush_size = int(l2_flush_size_mb) * 1024 * 1024
        buffer = torch.empty(l2_flush_size, device=l2_flush_device, dtype=torch.int8)

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    torch.cuda.synchronize()
    fn()  # Call once to exclude initial overhead
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(measurement_iters):
        if l2_flush:
            buffer.zero_()
        fn()
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
        if l2_flush:
            buffer.zero_()
        fn()
    torch.cuda.synchronize()

    # Actual run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.cuda.synchronize()
    for iter_idx in range(repeat_iters):
        if l2_flush:
            buffer.zero_()
        start_events[iter_idx].record()
        fn()
        end_events[iter_idx].record()

        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)

    # Synchronize once outside of the loop to avoid synchronization overhead
    torch.cuda.synchronize()
    measured_times = []
    for iter_idx in range(repeat_iters):
        measured_times.append(start_events[iter_idx].elapsed_time(end_events[iter_idx]))
    return measured_times


def bench_gpu_time_with_cudagraph(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    num_iters_within_graph: int = 10,
    l2_flush: bool = True,
    l2_flush_size_mb: int = 256,
    l2_flush_device: str = "cuda",
    sleep_after_run: bool = False,
):
    """
    Benchmark GPU time using by constructing CUDA graphs with kernel launch and then replaying the graph.
    Increasing the number of iterations within graph can amortize kernel launch latency to help
    obtain measurements close to GPU kernel time of fn().
    Can flush L2 cache and sleep after the run.

    Number of dry run and actual run iterations can be set by iteration count or time:
    - If dry_run_iters and repeat_iters are provided, provided iteration count will be used.
    - If dry_run_iters and repeat_iters are not provided, dry_run_time_ms and repeat_time_ms will be used.

    Returns an array of measured times so that the caller can compute statistics.

    Uses PyTorch's API to construt and use CUDA Graphs.
    Also see PyTorch's post on CUDA Graphs: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/

    Args:
        fn: Function to benchmark.
        dry_run_iters: Number of dry runs during which times does not count. If not provided, dry_run_time_ms will be used.
        repeat_iters: Number of iterations. If not provided, repeat_time_ms will be used.
        dry_run_time_ms: Time to run the dry run in milliseconds.
        repeat_time_ms: Time to run the repeat in milliseconds.
        num_iters_within_graph: Number of iterations to run within the graph.
        l2_flush: Whether to flush L2 cache.
        l2_flush_size_mb: Size of the L2 cache to flush.
        l2_flush_device: Device that needs to flush L2 cache.
        sleep_after_run: Whether to sleep after the run. Sleep time is dynamically set.

    Returns:
        measured_times: List of measured times.
    """

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    if l2_flush:
        l2_flush_size = int(l2_flush_size_mb) * 1024 * 1024
        buffer = torch.empty(l2_flush_size, device=l2_flush_device, dtype=torch.int8)

    # Warmup run
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)

    # Capture kernel in graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(num_iters_within_graph):
            fn()
    torch.cuda.synchronize()

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    start_event.record()
    for _ in range(measurement_iters):
        if l2_flush:
            buffer.zero_()
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
        if l2_flush:
            buffer.zero_()
        g.replay()
    torch.cuda.synchronize()

    # Actual run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.cuda.synchronize()
    for iter_idx in range(repeat_iters):
        if l2_flush:
            buffer.zero_()
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
