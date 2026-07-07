# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

import os

_AUTOTUNE_DISABLED = os.getenv('FLASHINFER_CUTILE_AUTOTUNE_DISABLED', '0') == '1'

# Module-level tune cache: (Q, M, N, K, transpose_a_int, transpose_b_int, dtype, device) -> (best_cfg, tuned_kernel)
_masked_bmm_tune_cache: dict = {}


@ct.kernel
def _masked_bmm_kernel(
    a,  # Input matrix A [Q, M, K] or [Q, K, M] if transpose_a
    b,  # Input matrix B [Q, K, N] or [Q, N, K] if transpose_b
    c,  # Output matrix C [Q, M, N]
    masked_m,  # Per-batch M mask [Q], int32
    TRANSPOSE_A: ct.Constant[int],  # Whether A is transposed (0 or 1)
    TRANSPOSE_B: ct.Constant[int],  # Whether B is transposed (0 or 1)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    cuTile kernel for masked batched matrix multiplication.

    Performs A @ B with per-batch M masking where:
    - A is batched [Q, M, K] or [Q, K, M] if transpose_a
    - B is batched [Q, K, N] or [Q, N, K] if transpose_b
    - masked_m is per-batch M mask [Q]
    - Output C is [Q, M, N]

    Uses persistent scheduling with static grid and GROUP_SIZE_M tile swizzling.
    """
    pid = ct.bid(0)

    zero_pad = ct.PaddingMode.ZERO

    # Compute num_k_tiles from tensor shape using ct.num_tiles
    # For non-transposed A: shape is [Q, M, K], we tile K (axis=2)
    # For transposed A: shape is [Q, K, M], we tile K (axis=1)
    if TRANSPOSE_A == 1:
        num_k_tiles = ct.num_tiles(a, axis=1, shape=(1, BLOCK_K, BLOCK_M))
    else:
        num_k_tiles = ct.num_tiles(a, axis=2, shape=(1, BLOCK_M, BLOCK_K))

    num_q = ct.num_tiles(c, axis=0, shape=(1, BLOCK_M, BLOCK_N))
    num_pid_m = ct.num_tiles(c, axis=1, shape=(1, BLOCK_M, BLOCK_N))
    num_pid_n = ct.num_tiles(c, axis=2, shape=(1, BLOCK_M, BLOCK_N))
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = num_q * tiles_per_batch
    num_programs = ct.num_blocks(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Persistent scheduling loop
    for current_pid in range(pid, total_tiles, num_programs):
        # Calculate pid_q, pid_m, pid_n with GROUP_SIZE_M swizzling
        pid_q = current_pid // tiles_per_batch
        pid_in_batch = current_pid % tiles_per_batch

        group_id = pid_in_batch // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_m = first_pid_m + (pid_in_batch % group_size_m_actual)
        pid_n = (pid_in_batch % num_pid_in_group) // group_size_m_actual

        # Load valid_m for this batch
        valid_m_tile = ct.load(masked_m, index=(pid_q,), shape=(1,))
        valid_m = valid_m_tile.item()

        # Only process if this tile is within valid M range
        if pid_m * BLOCK_M < valid_m:
            # Initialize accumulator
            acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            # K-loop for matrix multiplication using tile indices
            for k in range(num_k_tiles):
                # Load A block based on transpose_a flag
                # Using tile-index based loading (k is tile index, not element offset)
                if TRANSPOSE_A == 1:
                    # A is [Q, K, M], load [1, BLOCK_K, BLOCK_M] using tile indices
                    a_block_3d = ct.load(
                        a,
                        index=(pid_q, k, pid_m),  # tile indices
                        shape=(1, BLOCK_K, BLOCK_M),
                        order=(0, 1, 2),
                        padding_mode=zero_pad,
                    )
                    a_block_km = ct.reshape(a_block_3d, (BLOCK_K, BLOCK_M))
                    a_block = ct.permute(a_block_km, (1, 0))  # [BLOCK_M, BLOCK_K]
                else:
                    # A is [Q, M, K], load [1, BLOCK_M, BLOCK_K] using tile indices
                    a_block_3d = ct.load(
                        a,
                        index=(pid_q, pid_m, k),  # tile indices
                        shape=(1, BLOCK_M, BLOCK_K),
                        order=(0, 1, 2),
                        padding_mode=zero_pad,
                    )
                    a_block = ct.reshape(a_block_3d, (BLOCK_M, BLOCK_K))

                # Load B block based on transpose_b flag
                if TRANSPOSE_B == 1:
                    # B is [Q, N, K], load [1, BLOCK_N, BLOCK_K] using tile indices
                    b_block_3d = ct.load(
                        b,
                        index=(pid_q, pid_n, k),  # tile indices
                        shape=(1, BLOCK_N, BLOCK_K),
                        order=(0, 1, 2),
                        padding_mode=zero_pad,
                    )
                    b_block_nk = ct.reshape(b_block_3d, (BLOCK_N, BLOCK_K))
                    b_block = ct.permute(b_block_nk, (1, 0))  # [BLOCK_K, BLOCK_N]
                else:
                    # B is [Q, K, N], load [1, BLOCK_K, BLOCK_N] using tile indices
                    b_block_3d = ct.load(
                        b,
                        index=(pid_q, k, pid_n),  # tile indices
                        shape=(1, BLOCK_K, BLOCK_N),
                        order=(0, 1, 2),
                        padding_mode=zero_pad,
                    )
                    b_block = ct.reshape(b_block_3d, (BLOCK_K, BLOCK_N))

                # Matrix multiplication: A @ B
                acc = ct.mma(a_block, b_block, acc=acc)

            # Convert to output dtype and store
            c_block = ct.astype(acc, c.dtype)

            # Reshape to 3D for store [1, BLOCK_M, BLOCK_N]
            c_block_3d = ct.reshape(c_block, (1, BLOCK_M, BLOCK_N))

            # Store to output C [Q, M, N] using tile indices
            ct.store(
                c,
                index=(pid_q, pid_m, pid_n),  # tile indices
                tile=c_block_3d,
                order=(0, 1, 2),
            )


def _masked_bmm_autotune_configs():
    """
    Iterator of autotune configurations for masked BMM kernel.

    IMPORTANT: Focus tuning on num_ctas and occupancy as requested.
    - num_ctas: Number of CTAs in a CGA (valid: 1, 2, 4, 8, 16)
    - occupancy: Expected active CTAs per SM (range: 1-32)

    For GEMM-like kernels:
    - Higher num_ctas can improve L2 cache hit rate via CGA
    - Occupancy affects latency hiding vs register pressure tradeoff
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability[0] == 10:
        # B200 / GB200 (sm100 / sm103)
        for BM, BN in [
            (128, 128),
            (128, 256),
            (256, 128),
            (256, 256),
        ]:
            for BK in [64]:
                # Focus on num_ctas tuning: 1, 2, 4 are most common for GEMM
                for num_ctas in [1, 2, 4]:
                    # Focus on occupancy tuning: 1-4 for compute-bound GEMM
                    for occupancy in [1, 2, 4]:
                        yield SimpleNamespace(
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=8, num_ctas=num_ctas, occupancy=occupancy
                        )
    elif gpu_capability in [(12, 0), (12, 1)]:
        # RTX 5090 (sm120/sm121)
        for BM, BN in [
            (128, 128),
            (128, 256),
            (256, 128),
            (256, 256),
        ]:
            for BK in [64]:
                for num_ctas in [1, 2, 4]:
                    for occupancy in [1, 2, 4]:
                        yield SimpleNamespace(
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=8, num_ctas=num_ctas, occupancy=occupancy
                        )
    elif gpu_capability == (9, 0):
        # H100 (sm_90) - supports higher num_ctas values
        for BM, BN in [
            (128, 128),
            (128, 256),
            (256, 128),
            (256, 256),
        ]:
            for BK in [64]:
                # H100 can benefit from higher num_ctas
                for num_ctas in [1, 2, 4, 8]:
                    for occupancy in [1, 2, 4, 8]:
                        yield SimpleNamespace(
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=8, num_ctas=num_ctas, occupancy=occupancy
                        )
    else:
        # Default configurations
        for BM, BN in [
            (128, 128),
            (128, 256),
        ]:
            for BK in [64]:
                for num_ctas in [1, 2]:
                    for occupancy in [1, 2, 4]:
                        yield SimpleNamespace(
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=8, num_ctas=num_ctas, occupancy=occupancy
                        )


def _get_default_kernel_configs():
    """
    Get GPU-specific default kernel configs for non-autotune path.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability[0] == 10:
        # B200 / GB200 (sm100 / sm103)
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 2,
        }
    elif gpu_capability in [(12, 0), (12, 1)]:
        # RTX 5090 (sm120/sm121)
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 2,
        }
    elif gpu_capability == (9, 0):
        # H100 - higher num_ctas default
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 2,
            "occupancy": 1,
        }
    else:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
        }


def _masked_bmm_autotune(stream, a, b, c, masked_m, Q, M, N, transpose_a, transpose_b):
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    transpose_a_int = 1 if transpose_a else 0
    transpose_b_int = 1 if transpose_b else 0

    def args_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        BK = cfg.BLOCK_K
        GSM = cfg.GROUP_SIZE_M

        return (
            a,
            b,
            c,
            masked_m,
            transpose_a_int,
            transpose_b_int,
            BM,
            BN,
            BK,
            GSM,
        )

    def grid_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        num_pid_m = ct.cdiv(M, BM)
        num_pid_n = ct.cdiv(N, BN)
        tiles_per_batch = num_pid_m * num_pid_n
        total_tiles = tiles_per_batch * Q
        num_programs = min(NUM_SMS // cfg.num_ctas, total_tiles) * cfg.occupancy
        return (num_programs, 1, 1)

    def hints_fn(cfg):
        return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

    K = a.shape[1] if transpose_a else a.shape[2]
    cache_key = (Q, M, N, K, transpose_a_int, transpose_b_int, a.dtype, str(a.device))
    if cache_key not in _masked_bmm_tune_cache:
        result = exhaustive_search(
            list(_masked_bmm_autotune_configs()),
            stream,
            grid_fn,
            _masked_bmm_kernel,
            args_fn,
            hints_fn,
        )
        best_cfg = result.best.config
        _masked_bmm_tune_cache[cache_key] = (
            best_cfg,
            _masked_bmm_kernel.replace_hints(**hints_fn(best_cfg)),
        )
    best_cfg, tuned_kernel = _masked_bmm_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


def masked_bmm(
    a,
    b,
    masked_m,
    transpose_a=False,
    transpose_b=False,
    static_persistent=None,
    **kwargs,
):
    """
    cuTile implementation of masked batched matrix multiplication.

    Performs A @ B with per-batch M masking where:
    - A is batched [Q, M, K] or [Q, K, M] if transpose_a
    - B is batched [Q, K, N] or [Q, N, K] if transpose_b
    - masked_m is per-batch M mask [Q]

    Args:
        a: Input matrix A, batched [Q, M, K] or [Q, K, M] if transpose_a
        b: Input matrix B, batched [Q, K, N] or [Q, N, K] if transpose_b
        masked_m: Per-batch M mask tensor [Q]
        transpose_a: Whether A is transposed
        transpose_b: Whether B is transposed
        static_persistent: Whether to use static persistent (unused, for API compat)

    Returns:
        Output tensor C [Q, M, N]
    """
    # Get dimensions from input tensors
    if transpose_a:
        Q_A, K_A, M = a.shape
    else:
        Q_A, M, K_A = a.shape

    if transpose_b:
        Q_B, N, K_B = b.shape
    else:
        Q_B, K_B, N = b.shape

    assert K_A == K_B, "incompatible dimensions"
    assert Q_A == Q_B, "incompatible dimensions"
    Q = Q_A

    assert a.is_contiguous(), "A matrix must be contiguous"
    assert b.is_contiguous(), "B matrix must be contiguous"
    assert masked_m.is_contiguous(), "Masked matrix must be contiguous"
    assert masked_m.shape.numel() == Q, "Masked matrix must have the same shape as the number of batches"
    c = torch.empty((Q, M, N), device=a.device, dtype=a.dtype)

    enable_autotune = not _AUTOTUNE_DISABLED

    if enable_autotune:
        _masked_bmm_autotune(torch.cuda.current_stream(), a, b, c, masked_m, Q, M, N, transpose_a, transpose_b)
    else:
        default_configs = _get_default_kernel_configs()
        kernel_configs = {**default_configs, **(kwargs.get("kernel_configs") or {})}

        BLOCK_M = kernel_configs.get("BLOCK_M")
        BLOCK_N = kernel_configs.get("BLOCK_N")
        BLOCK_K = kernel_configs.get("BLOCK_K")
        GROUP_SIZE_M = kernel_configs.get("GROUP_SIZE_M", 8)
        num_ctas = kernel_configs.get("num_ctas", 1)
        occupancy = kernel_configs.get("occupancy", 1)

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid_m = ct.cdiv(M, BLOCK_M)
        num_pid_n = ct.cdiv(N, BLOCK_N)
        tiles_per_batch = num_pid_m * num_pid_n
        total_tiles = tiles_per_batch * Q
        num_programs = min(NUM_SMS // num_ctas, total_tiles) * occupancy

        grid = (num_programs, 1, 1)

        transpose_a_int = 1 if transpose_a else 0
        transpose_b_int = 1 if transpose_b else 0

        hints = {}
        if num_ctas is not None:
            hints["num_ctas"] = num_ctas
        if occupancy is not None:
            hints["occupancy"] = occupancy
        kernel = _masked_bmm_kernel.replace_hints(**hints) if hints else _masked_bmm_kernel

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel,
            (
                a,
                b,
                c,
                masked_m,
                transpose_a_int,
                transpose_b_int,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
            ),
        )

    return c
