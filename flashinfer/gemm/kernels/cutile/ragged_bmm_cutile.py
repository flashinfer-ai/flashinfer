# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from ....cutile.cutile_common import cached_replace_hints

import os

_AUTOTUNE_DISABLED = os.getenv("FLASHINFER_CUTILE_AUTOTUNE_DISABLED", "0") == "1"

# Module-level tune caches for standard and swap_ab ragged BMM
_ragged_bmm_standard_tune_cache: dict = {}
_ragged_bmm_swap_ab_tune_cache: dict = {}


@ct.kernel
def _ragged_bmm_kernel(
    a,  # Input matrix A [total_m, K] or [K, total_m] if transpose_a
    b,  # Input matrix B [Q, N, K] or [Q, K, N]
    c,  # Output matrix C [total_m, N]
    m_indptr,  # Segment offsets [Q+1], flattened 1D
    q,  # Number of batches
    max_m,  # Host-side max segment size hint (kept for autotune cache key)
    max_m_device,  # 1-element int32 tensor (shape (1,)) — device-side ground truth for max(valid_m)
    n,  # Output N dimension
    TRANSPOSE_A: ct.Constant[int],  # Whether A is transposed (0 or 1)
    TRANSPOSE_B: ct.Constant[int],  # Whether B is transposed (0 or 1)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    cuTile kernel for ragged batched matrix multiplication.

    Performs A @ B^T or A @ B where:
    - A is flattened with segment offsets (m_indptr defines boundaries)
    - B is batched [Q, N, K] or [Q, K, N]
    - Output C is [total_m, N]

    Uses persistent scheduling with static grid and GROUP_SIZE_M tile swizzling.
    Uses Array.slice + TMA (ct.load/ct.store) for A and C access.

    Defense-in-depth: the per-tile loop bound is computed from the device-side
    `max_m_device` rather than the host-side `max_m`. This prevents silent output corruption when the caller passes
    too small a host-side max_m hint (rows beyond cdiv(max_m, BLOCK_M)*BLOCK_M
    would otherwise never be processed).
    This matches the NVT triton kernel's defensive semantic.
    """
    pid = ct.bid(0)

    if TRANSPOSE_A == 1:
        num_k_tiles = ct.num_tiles(a, axis=0, shape=(BLOCK_K, BLOCK_M))
    else:
        num_k_tiles = ct.num_tiles(a, axis=1, shape=(BLOCK_M, BLOCK_K))
    # Override host max_m with device truth (see docstring).
    max_m_runtime = ct.load(max_m_device, index=(0,), shape=(1,)).item()
    num_pid_m = ct.cdiv(max_m_runtime, BLOCK_M)
    num_pid_n = ct.cdiv(n, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * q
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

        # Load segment boundaries using ct.load with dynamic index
        m_start_tile = ct.load(m_indptr, index=(pid_q,), shape=(1,))
        m_start = m_start_tile.item()
        m_end_tile = ct.load(m_indptr, index=(pid_q + 1,), shape=(1,))
        m_end = m_end_tile.item()
        valid_m = m_end - m_start

        # Only process if this tile is within valid M range
        if pid_m * BLOCK_M < valid_m:
            # Create sliced views for A and C using Array.slice
            # This enables TMA (ct.load/ct.store) on ragged data
            if TRANSPOSE_A == 1:
                # A is [K, total_m], slice axis 1 for M dimension
                Ai = a.slice(axis=1, start=m_start, stop=m_end)  # shape: (K, valid_m)
            else:
                # A is [total_m, K], slice axis 0 for M dimension
                Ai = a.slice(axis=0, start=m_start, stop=m_end)  # shape: (valid_m, K)

            # Slice C along axis 0 for M dimension
            Ci = c.slice(axis=0, start=m_start, stop=m_end)  # shape: (valid_m, N)

            # Initialize accumulator
            dot_acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            # K-loop for matrix multiplication using TMA for A
            for k in range(num_k_tiles):
                # Load A block based on transpose_a flag using ct.load on sliced array
                if TRANSPOSE_A == 1:
                    # Ai is [K, valid_m], load [BLOCK_K, BLOCK_M] then transpose
                    a_block_kt = ct.load(
                        Ai,
                        index=(k, pid_m),
                        shape=(BLOCK_K, BLOCK_M),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    a_block = ct.permute(a_block_kt, (1, 0))  # [BLOCK_M, BLOCK_K]
                else:
                    # Ai is [valid_m, K], load [BLOCK_M, BLOCK_K]
                    a_block = ct.load(
                        Ai,
                        index=(pid_m, k),
                        shape=(BLOCK_M, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )

                # Load B block based on transpose_b flag using tile indices
                if TRANSPOSE_B == 1:
                    # B is [Q, N, K], load [1, BLOCK_N, BLOCK_K]
                    b_block_3d = ct.load(
                        b,
                        index=(pid_q, pid_n, k),
                        shape=(1, BLOCK_N, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_block = ct.reshape(b_block_3d, (BLOCK_N, BLOCK_K))
                    # Transpose B: [BLOCK_N, BLOCK_K] -> [BLOCK_K, BLOCK_N]
                    b_block_t = ct.permute(b_block, (1, 0))
                else:
                    # B is [Q, K, N], load [1, BLOCK_K, BLOCK_N]
                    b_block_3d = ct.load(
                        b,
                        index=(pid_q, k, pid_n),
                        shape=(1, BLOCK_K, BLOCK_N),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_block_t = ct.reshape(b_block_3d, (BLOCK_K, BLOCK_N))

                # Matrix multiplication: A @ B
                dot_acc = ct.mma(a_block, b_block_t, acc=dot_acc)

            # Convert to output dtype
            c_block = ct.astype(dot_acc, c.dtype)

            # Store to output C using ct.store on sliced array
            # Ci is [valid_m, N], store at tile index (pid_m, pid_n)
            # padding_mode handles partial tiles at segment boundaries
            ct.store(Ci, index=(pid_m, pid_n), tile=c_block)


@ct.kernel
def _ragged_bmm_swap_ab_kernel(
    a,  # Input matrix A [total_m, K] or [K, total_m] if transpose_a
    b,  # Input matrix B [Q, N, K] or [Q, K, N]
    c,  # Output matrix C [total_m, N]
    m_indptr,  # Segment offsets [Q+1], flattened 1D
    q,  # Number of batches
    max_m,  # Host-side max segment size hint (kept for autotune cache key)
    max_m_device,  # 1-element int32 tensor (shape (1,)) — device-side ground truth for max(valid_m)
    n,  # Output N dimension
    TRANSPOSE_A: ct.Constant[int],  # Whether A is transposed (0 or 1)
    TRANSPOSE_B: ct.Constant[int],  # Whether B is transposed (0 or 1)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    cuTile kernel for ragged batched matrix multiplication with swap_ab optimization.

    Uses swapped accumulator layout (BLOCK_N, BLOCK_M) for better performance
    when M dimension is small. Equivalent to: dot(B^T.T, A.T).T = A @ B^T

    Uses Array.slice + TMA (ct.load/ct.store) for A and C access.

    Defense-in-depth: same as `_ragged_bmm_kernel` — the per-tile loop bound
    is computed from `max_m_device` (device truth), not the host hint.
    """
    pid = ct.bid(0)

    if TRANSPOSE_A == 1:
        num_k_tiles = ct.num_tiles(a, axis=0, shape=(BLOCK_K, BLOCK_M))
    else:
        num_k_tiles = ct.num_tiles(a, axis=1, shape=(BLOCK_M, BLOCK_K))
    max_m_runtime = ct.load(max_m_device, index=(0,), shape=(1,)).item()
    num_pid_m = ct.cdiv(max_m_runtime, BLOCK_M)
    num_pid_n = ct.cdiv(n, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * q
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

        # Load segment boundaries using ct.load with dynamic index
        m_start_tile = ct.load(m_indptr, index=(pid_q,), shape=(1,))
        m_start = m_start_tile.item()
        m_end_tile = ct.load(m_indptr, index=(pid_q + 1,), shape=(1,))
        m_end = m_end_tile.item()
        valid_m = m_end - m_start

        # Only process if this tile is within valid M range
        if pid_m * BLOCK_M < valid_m:
            # Create sliced views for A and C using Array.slice
            # This enables TMA (ct.load/ct.store) on ragged data
            if TRANSPOSE_A == 1:
                # A is [K, total_m], slice axis 1 for M dimension
                Ai = a.slice(axis=1, start=m_start, stop=m_end)  # shape: (K, valid_m)
            else:
                # A is [total_m, K], slice axis 0 for M dimension
                Ai = a.slice(axis=0, start=m_start, stop=m_end)  # shape: (valid_m, K)

            # Slice C along axis 0 for M dimension
            Ci = c.slice(axis=0, start=m_start, stop=m_end)  # shape: (valid_m, N)

            # Initialize accumulator with swapped dimensions [BLOCK_N, BLOCK_M]
            dot_acc = ct.full((BLOCK_N, BLOCK_M), 0.0, dtype=ct.float32)

            # K-loop for matrix multiplication using TMA for A
            for k in range(num_k_tiles):
                # Load A block based on transpose_a flag using ct.load on sliced array
                if TRANSPOSE_A == 1:
                    # Ai is [K, valid_m], load [BLOCK_K, BLOCK_M] then transpose
                    a_block_kt = ct.load(
                        Ai,
                        index=(k, pid_m),
                        shape=(BLOCK_K, BLOCK_M),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    a_block = ct.permute(a_block_kt, (1, 0))  # [BLOCK_M, BLOCK_K]
                else:
                    # Ai is [valid_m, K], load [BLOCK_M, BLOCK_K]
                    a_block = ct.load(
                        Ai,
                        index=(pid_m, k),
                        shape=(BLOCK_M, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )

                # Load B block based on transpose_b flag using tile indices
                if TRANSPOSE_B == 1:
                    # B is [Q, N, K], load [1, BLOCK_N, BLOCK_K]
                    b_block_3d = ct.load(
                        b,
                        index=(pid_q, pid_n, k),
                        shape=(1, BLOCK_N, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_block = ct.reshape(b_block_3d, (BLOCK_N, BLOCK_K))
                else:
                    # B is [Q, K, N], load [1, BLOCK_K, BLOCK_N]
                    b_block_3d = ct.load(
                        b,
                        index=(pid_q, k, pid_n),
                        shape=(1, BLOCK_K, BLOCK_N),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_block_kn = ct.reshape(b_block_3d, (BLOCK_K, BLOCK_N))
                    b_block = ct.permute(b_block_kn, (1, 0))  # [BLOCK_N, BLOCK_K]

                # For swap_ab: compute B @ A^T = [BLOCK_N, BLOCK_K] @ [BLOCK_K, BLOCK_M]
                a_block_t = ct.permute(a_block, (1, 0))  # [BLOCK_K, BLOCK_M]

                # Matrix multiplication: B @ A^T
                dot_acc = ct.mma(b_block, a_block_t, acc=dot_acc)

            # Transpose back: [BLOCK_N, BLOCK_M] -> [BLOCK_M, BLOCK_N]
            acc_transposed = ct.permute(dot_acc, (1, 0))

            # Convert to output dtype
            c_block = ct.astype(acc_transposed, c.dtype)

            # Store to output C using ct.store on sliced array
            # Ci is [valid_m, N], store at tile index (pid_m, pid_n)
            # padding_mode handles partial tiles at segment boundaries
            ct.store(Ci, index=(pid_m, pid_n), tile=c_block)


# OPTIMIZED: Expanded search space matching Triton's configurations with extended occupancy range.
def _ragged_bmm_autotune_configs_standard():
    """
    Iterator of autotune configurations for standard (non-swap_ab) kernel.

    with extended occupancy range for better workload coverage.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability[0] == 10:
        # SM100 / SM103 - expanded configs
        for BM, BN in [
            (256, 256),
            (128, 256),
            (128, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )
    elif gpu_capability in [(12, 0), (12, 1)]:
        # RTX 5090 (sm120/sm121) - expanded configs
        for BM, BN in [
            (256, 256),
            (128, 256),
            (128, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )
    elif gpu_capability == (9, 0):
        # H100 (sm_90) - expanded configs
        for BM, BN in [
            (256, 256),
            (128, 256),
            (128, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4, 8]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )
    else:
        # Default configurations - expanded
        for BM, BN in [
            (128, 256),
            (128, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )


def _ragged_bmm_autotune_configs_swap_ab():
    """
    Iterator of autotune configurations for swap_ab kernel.
    Used when M dimension is small relative to N.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability[0] == 10:
        # SM100 / SM103
        for BM, BN in [
            (64, 256),
            (64, 128),
            (32, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )
    elif gpu_capability in [(12, 0), (12, 1)]:
        # RTX 5090 (sm120/sm121)
        for BM, BN in [
            (64, 256),
            (64, 128),
            (32, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )
    elif gpu_capability == (9, 0):
        # H100 (sm_90)
        for BM, BN in [
            (64, 256),
            (64, 128),
            (32, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4, 8]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )
    else:
        # Default configurations
        for BM, BN in [
            (64, 128),
            (32, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                    )


def _get_default_kernel_configs():
    """
    Get GPU-specific default kernel configs for non-autotune path.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability[0] == 10:
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "swap_ab": False,
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
            "swap_ab": False,
            "num_ctas": 1,
            "occupancy": 2,
        }
    elif gpu_capability == (9, 0):
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "swap_ab": False,
            "num_ctas": 1,
            "occupancy": 2,
        }
    else:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "swap_ab": False,
            "num_ctas": 1,
            "occupancy": 2,
        }


def _ragged_bmm_autotune_standard(
    stream,
    a,
    b,
    c,
    m_indptr,
    Q,
    max_m,
    max_m_device,
    N,
    total_m,
    transpose_a,
    transpose_b,
):
    """
    Autotuned launch for standard ragged BMM kernel.
    """
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
            m_indptr,
            Q,
            max_m,
            max_m_device,
            N,
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
        # Size the grid from total_m (the true row-count upper bound), not the
        # host max_m hint: a stale/zero max_m would yield 0 tiles -> empty grid
        # -> unwritten output. The kernel still reads its per-tile bound from
        # max_m_device.
        num_pid_m = ct.cdiv(total_m, BM)
        num_pid_n = ct.cdiv(N, BN)
        tiles_per_batch = num_pid_m * num_pid_n
        total_tiles = tiles_per_batch * Q
        num_programs = min(NUM_SMS // cfg.num_ctas, total_tiles) * cfg.occupancy
        return (num_programs, 1, 1)

    def hints_fn(cfg):
        return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

    cache_key = (
        Q,
        max_m,
        N,
        total_m,
        transpose_a_int,
        transpose_b_int,
        a.dtype,
        str(a.device),
    )
    if cache_key not in _ragged_bmm_standard_tune_cache:
        result = exhaustive_search(
            list(_ragged_bmm_autotune_configs_standard()),
            stream,
            grid_fn,
            _ragged_bmm_kernel,
            args_fn,
            hints_fn,
        )
        best_cfg = result.best.config
        _ragged_bmm_standard_tune_cache[cache_key] = (
            best_cfg,
            _ragged_bmm_kernel.replace_hints(**hints_fn(best_cfg)),
        )
    best_cfg, tuned_kernel = _ragged_bmm_standard_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


def _ragged_bmm_autotune_swap_ab(
    stream,
    a,
    b,
    c,
    m_indptr,
    Q,
    max_m,
    max_m_device,
    N,
    total_m,
    transpose_a,
    transpose_b,
):
    """
    Autotuned launch for swap_ab ragged BMM kernel.
    """
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
            m_indptr,
            Q,
            max_m,
            max_m_device,
            N,
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
        # Size the grid from total_m (the true row-count upper bound), not the
        # host max_m hint: a stale/zero max_m would yield 0 tiles -> empty grid
        # -> unwritten output. The kernel still reads its per-tile bound from
        # max_m_device.
        num_pid_m = ct.cdiv(total_m, BM)
        num_pid_n = ct.cdiv(N, BN)
        tiles_per_batch = num_pid_m * num_pid_n
        total_tiles = tiles_per_batch * Q
        num_programs = min(NUM_SMS // cfg.num_ctas, total_tiles) * cfg.occupancy
        return (num_programs, 1, 1)

    def hints_fn(cfg):
        return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

    swap_cache_key = (
        Q,
        max_m,
        N,
        total_m,
        transpose_a_int,
        transpose_b_int,
        a.dtype,
        str(a.device),
    )
    if swap_cache_key not in _ragged_bmm_swap_ab_tune_cache:
        result = exhaustive_search(
            list(_ragged_bmm_autotune_configs_swap_ab()),
            stream,
            grid_fn,
            _ragged_bmm_swap_ab_kernel,
            args_fn,
            hints_fn,
        )
        best_cfg = result.best.config
        _ragged_bmm_swap_ab_tune_cache[swap_cache_key] = (
            best_cfg,
            _ragged_bmm_swap_ab_kernel.replace_hints(**hints_fn(best_cfg)),
        )
    best_cfg, tuned_kernel = _ragged_bmm_swap_ab_tune_cache[swap_cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


def ragged_bmm(
    a,
    b,
    m_indptr,
    max_m,
    max_m_device=None,
    transpose_a=False,
    transpose_b=True,
    out_dtype=None,
    **kwargs,
):
    """
    cuTile implementation of ragged BMM with non-even M segments.

    Matrix A is flattened with m_indptr defining the boundaries.
    Performs A @ B (or A @ B^T if transpose_b=True) where B is batched.

    Note: For cuTile implementation, segment offsets should ideally be
    multiples of BLOCK_M for optimal performance and correctness.

    Args:
        a: Input matrix A, flattened [total_m, K] or [K, total_m] if transpose_a
        b: Input matrix B, batched [Q, N, K] or [Q, K, N] if not transpose_b
        m_indptr: Segment offsets tensor [Q+1]
        max_m: Host-side maximum segment size hint (used for grid sizing and
            autotune cache key). Should be >= max per-batch valid_m.
        max_m_device: Optional [1]-shape int tensor with the device-side ground
            truth for max(valid_m). When provided, the kernel uses this value
            for its persistent-loop bound — making the kernel robust to a
            host-side max_m underestimate. When None, a fallback tensor is
            materialized from `max_m`.
        transpose_a: Whether A is transposed
        transpose_b: Whether B is transposed
        out_dtype: Output dtype

    Returns:
        Output tensor C [total_m, N]
    """
    # Get dimensions from flattened matrix a
    if transpose_a:
        K, total_m = a.shape
    else:
        total_m, K = a.shape

    if transpose_b:
        Q, N, K_B = b.shape
    else:
        Q, K_B, N = b.shape

    assert K == K_B, "incompatible dimensions"
    assert m_indptr.shape[0] == Q + 1, "m_indptr must have Q+1 elements"
    assert a.is_contiguous(), "A matrix must be contiguous"
    assert b.is_contiguous(), "B matrix must be contiguous"
    assert m_indptr.is_contiguous(), "m_indptr must be contiguous"

    # Determine output dtype
    if out_dtype is None:
        out_dtype = a.dtype
    c = torch.empty((total_m, N), device=a.device, dtype=out_dtype)

    # Materialize fallback max_m_device if the caller didn't pass one. The
    # kernel always reads its grid bound from a device tensor (defense-in-depth),
    # so we keep the call sites uniform.
    # Note: this mirrors the NVT triton kernel, which always reads its grid
    # bound from a device tensor.
    if max_m_device is None:
        max_m_device = torch.tensor([max_m], dtype=torch.int32, device=a.device)

    # Check if autotune is enabled
    enable_autotune = not _AUTOTUNE_DISABLED

    # Decide whether to use swap_ab based on M vs N ratio
    # swap_ab is beneficial when M is small relative to N
    use_swap_ab = max_m <= 128 and N >= 256

    if enable_autotune:
        if use_swap_ab:
            _ragged_bmm_autotune_swap_ab(
                torch.cuda.current_stream(a.device),
                a,
                b,
                c,
                m_indptr,
                Q,
                max_m,
                max_m_device,
                N,
                total_m,
                transpose_a,
                transpose_b,
            )
        else:
            _ragged_bmm_autotune_standard(
                torch.cuda.current_stream(a.device),
                a,
                b,
                c,
                m_indptr,
                Q,
                max_m,
                max_m_device,
                N,
                total_m,
                transpose_a,
                transpose_b,
            )
    else:
        # Use fixed default configs
        default_configs = _get_default_kernel_configs()
        kernel_configs = {**default_configs, **(kwargs.get("kernel_configs") or {})}

        BLOCK_M = kernel_configs.get("BLOCK_M")
        BLOCK_N = kernel_configs.get("BLOCK_N")
        BLOCK_K = kernel_configs.get("BLOCK_K")
        GROUP_SIZE_M = kernel_configs.get("GROUP_SIZE_M", 8)
        swap_ab = kernel_configs.get("swap_ab", False)
        num_ctas = kernel_configs.get("num_ctas", 1)
        occupancy = kernel_configs.get("occupancy", 2)

        # Calculate grid size for persistent scheduling
        NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
        # See grid_fn note: size from total_m, not the host max_m hint.
        num_pid_m = ct.cdiv(total_m, BLOCK_M)
        num_pid_n = ct.cdiv(N, BLOCK_N)
        tiles_per_batch = num_pid_m * num_pid_n
        total_tiles = tiles_per_batch * Q
        num_programs = min(NUM_SMS // num_ctas, total_tiles) * occupancy

        # 1D grid for persistent scheduling
        grid = (num_programs, 1, 1)

        # Convert boolean to int for ct.Constant
        transpose_a_int = 1 if transpose_a else 0
        transpose_b_int = 1 if transpose_b else 0

        # Select kernel based on swap_ab
        kernel_fn = _ragged_bmm_swap_ab_kernel if swap_ab else _ragged_bmm_kernel

        hints = {}
        if num_ctas is not None:
            hints["num_ctas"] = num_ctas
        if occupancy is not None:
            hints["occupancy"] = occupancy
        kernel = cached_replace_hints(kernel_fn, **hints) if hints else kernel_fn

        ct.launch(
            torch.cuda.current_stream(a.device),
            grid,
            kernel,
            (
                a,
                b,
                c,
                m_indptr,
                Q,
                max_m,
                max_m_device,
                N,
                transpose_a_int,
                transpose_b_int,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
            ),
        )

    return c
