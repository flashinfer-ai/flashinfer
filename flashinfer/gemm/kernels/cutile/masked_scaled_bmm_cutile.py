# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile._stub import mma_scaled
from cuda.tile.tune import exhaustive_search

from ....cutile.cutile_common import cached_replace_hints

import os

_AUTOTUNE_DISABLED = os.getenv("FLASHINFER_CUTILE_AUTOTUNE_DISABLED", "0") == "1"

# Module-level per-device cache for the max_m 1-element buffer, to avoid
# repeated torch.zeros allocations when the op is called in a hot loop.
_max_m_cache: dict = {}

# Module-level tune cache: (Q, M, N, K_A, K_B, ELEM_PER_BYTE_A, VEC_SIZE,
# MIXED_PREC, dtype, device) -> (best_cfg, tuned_kernel)
_masked_scaled_bmm_tune_cache: dict = {}


@ct.kernel
def _masked_m_max_kernel(
    masked_m,
    out,
    Q: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    """Single-program cuTile kernel to compute max(masked_m[0:Q])."""
    # Load all Q elements (with zero-padding if BLOCK > Q)
    vals = ct.load(
        masked_m, index=(0,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO
    )
    max_val = ct.max(vals, axis=0)
    ct.store(out, index=(0,), tile=ct.reshape(max_val, (1,)))


def _masked_m_max_device_cutile(masked_m: torch.Tensor) -> torch.Tensor:
    """
    Returns a 1-element int32 CUDA tensor containing max(masked_m).
    Implemented via a small cuTile kernel to keep the reduction on-device.
    """
    device = masked_m.device
    Q = int(masked_m.numel())

    max_m_buf = _max_m_cache.get(device)
    if max_m_buf is None:
        max_m_buf = torch.zeros((1,), device=device, dtype=torch.int32)
        _max_m_cache[device] = max_m_buf
    else:
        max_m_buf.zero_()

    if Q == 0:
        return max_m_buf

    # Pick BLOCK as next power of 2 >= Q, capped at 1024
    BLOCK = (1 << max(0, (Q - 1)).bit_length()) if Q > 1 else 1
    BLOCK = max(BLOCK, 32)  # Minimum block size for efficiency
    BLOCK = min(BLOCK, 1024)

    if Q > BLOCK:
        # For large Q, fall back to torch.max on device (still async, no CPU sync).
        # torch.max requires out[0].dtype to match input.dtype, so cast masked_m
        # to int32 first if needed.
        src = masked_m if masked_m.dtype == torch.int32 else masked_m.to(torch.int32)
        torch.max(
            src,
            dim=0,
            out=(max_m_buf, torch.empty(1, device=device, dtype=torch.int64)),
        )
        return max_m_buf

    ct.launch(
        torch.cuda.current_stream(),
        (1, 1, 1),
        _masked_m_max_kernel,
        (masked_m, max_m_buf, Q, BLOCK),
    )
    return max_m_buf


@ct.kernel
def _masked_scaled_bmm_kernel(
    a,
    b,
    a_scale,  # Raw 5D: [Q, rm, rk, 2, 256]
    b_scale,  # Raw 5D: [Q, rn, rk, 2, 256]
    masked_m,
    c,
    max_m,
    q,
    m,
    n,
    ELEM_PER_BYTE_A: ct.Constant[int],
    VEC_SIZE: ct.Constant[int],
    SCALES_PER_BLOCK_K: ct.Constant[int],
    MIXED_PREC: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    SCALE_REP_M: ct.Constant[int],  # BLOCK_M // 128
    SCALE_REP_K: ct.Constant[int],  # BLOCK_K // VEC_SIZE // 4
):
    """
    cuTile kernel for masked block-scaled batched matrix multiplication.

    Performs C[q] = (A[q] * A_scale[q]) @ (B[q] * B_scale[q])^T where:
    - A is batched FP8/FP4 [Q, M, K_A]
    - B is batched FP8/FP4 [Q, N, K_B]
    - A_scale is raw 5D [Q, rm, rk, 2, 256] (MX hardware swizzle format)
    - B_scale is raw 5D [Q, rn, rk, 2, 256] (MX hardware swizzle format)
    - masked_m is [Q] — per-batch valid M count
    - max_m is [1] int32 — max(masked_m), computed on GPU
    - Output C is [Q, M, N]

    In-kernel scale unswizzle: loads 5D packed scales via TMA and unswizzles
    in-register using reshape+permute+reshape (same as Triton's approach).
    Uses persistent scheduling with dynamic grid based on max(masked_m).
    Uses mma_scaled for hardware-accelerated scaled MMA.
    """
    pid = ct.bid(0)

    # Load max(masked_m) from GPU tensor and compute dynamic scheduling params
    max_m_tile = ct.load(max_m, index=(0,), shape=(1,))
    max_m_val = max_m_tile.item()
    M_eff = ct.minimum(max_m_val, m)
    num_pid_m = ct.cdiv(M_eff, BLOCK_M)
    num_pid_n = ct.cdiv(n, BLOCK_N)
    num_programs = ct.num_blocks(0)
    if ELEM_PER_BYTE_A == 2:
        num_k_tiles = ct.num_tiles(a, axis=2, shape=(1, BLOCK_M, BLOCK_K // 2))
    else:
        num_k_tiles = ct.num_tiles(a, axis=2, shape=(1, BLOCK_M, BLOCK_K))
    tiles_per_batch = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    total_tiles = tiles_per_batch * q
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

        # Load per-batch valid M count
        valid_m_tile = ct.load(masked_m, index=(pid_q,), shape=(1,))
        valid_m = valid_m_tile.item()

        # Only process if this tile is within valid M range
        if pid_m * BLOCK_M < valid_m:
            acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            for k in range(num_k_tiles):
                if MIXED_PREC == 1:
                    # Mode 3: Mixed FP8×FP4 — A is fp8, B is fp4
                    # mma_scaled requires both operands to have the same dtype.
                    # Triton's tl.dot_scaled supports cross-precision (e4m3 × e2m1)
                    # natively, but mma_scaled does not. We downcast A from fp8
                    # to fp4 as a workaround, which may lose precision vs Triton.
                    # Load A as float8_e4m3fn [BLOCK_M, BLOCK_K]
                    a_block_3d = ct.load(
                        a,
                        index=(pid_q, pid_m, k),
                        shape=(1, BLOCK_M, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    a_fp8 = ct.reshape(a_block_3d, (BLOCK_M, BLOCK_K))
                    # Cast fp8 -> fp4 so both operands match for mma_scaled.
                    a_block = ct.astype(a_fp8, ct.float4_e2m1fn)

                    # Load B as uint8 bytes [BLOCK_N, BLOCK_K // 2], unpack to fp4
                    b_bytes_3d = ct.load(
                        b,
                        index=(pid_q, pid_n, k),
                        shape=(1, BLOCK_N, BLOCK_K // 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_bytes = ct.reshape(b_bytes_3d, (BLOCK_N, BLOCK_K // 2))
                    b_flat = ct.reshape(b_bytes, (-1,))
                    b_nk = ct.reshape(
                        ct.unpack_from_bytes(b_flat, ct.float4_e2m1fn),
                        (BLOCK_N, BLOCK_K),
                    )
                    b_block = ct.permute(b_nk, (1, 0))  # [BLOCK_K, BLOCK_N]

                elif ELEM_PER_BYTE_A == 2:
                    # Mode 2: FP4×FP4 — both A and B are fp4
                    a_bytes_3d = ct.load(
                        a,
                        index=(pid_q, pid_m, k),
                        shape=(1, BLOCK_M, BLOCK_K // 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    a_bytes = ct.reshape(a_bytes_3d, (BLOCK_M, BLOCK_K // 2))
                    a_flat = ct.reshape(a_bytes, (-1,))
                    a_block = ct.reshape(
                        ct.unpack_from_bytes(a_flat, ct.float4_e2m1fn),
                        (BLOCK_M, BLOCK_K),
                    )

                    b_bytes_3d = ct.load(
                        b,
                        index=(pid_q, pid_n, k),
                        shape=(1, BLOCK_N, BLOCK_K // 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_bytes = ct.reshape(b_bytes_3d, (BLOCK_N, BLOCK_K // 2))
                    b_flat = ct.reshape(b_bytes, (-1,))
                    b_nk = ct.reshape(
                        ct.unpack_from_bytes(b_flat, ct.float4_e2m1fn),
                        (BLOCK_N, BLOCK_K),
                    )
                    b_block = ct.permute(b_nk, (1, 0))

                else:
                    # Mode 1: FP8×FP8 — both A and B are fp8
                    a_block_3d = ct.load(
                        a,
                        index=(pid_q, pid_m, k),
                        shape=(1, BLOCK_M, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    a_block = ct.reshape(a_block_3d, (BLOCK_M, BLOCK_K))

                    b_block_3d = ct.load(
                        b,
                        index=(pid_q, pid_n, k),
                        shape=(1, BLOCK_N, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_nk = ct.reshape(b_block_3d, (BLOCK_N, BLOCK_K))
                    b_block = ct.permute(b_nk, (1, 0))

                # In-kernel scale unswizzle: load raw 5D and reshape/permute in-register
                # a_scale is raw [Q, rm, rk, 2, 256] — load block and unswizzle
                # Load shape: (1, SCALE_REP_M, SCALE_REP_K, 2, 256)
                a_scale_5d = ct.load(
                    a_scale,
                    index=(pid_q, pid_m, k, 0, 0),
                    shape=(1, SCALE_REP_M, SCALE_REP_K, 2, 256),
                    padding_mode=ct.PaddingMode.ZERO,
                )
                # Unswizzle: reshape to 6D, permute, reshape to 2D
                # (1, SCALE_REP_M, SCALE_REP_K, 2, 256) -> (1, SCALE_REP_M, SCALE_REP_K, 32, 4, 4)
                a_scale_6d = ct.reshape(
                    a_scale_5d, (1, SCALE_REP_M, SCALE_REP_K, 32, 4, 4)
                )
                # Permute (0,1,4,3,2,5) -> (1, SCALE_REP_M, 4, 32, SCALE_REP_K, 4)
                a_scale_perm = ct.permute(a_scale_6d, (0, 1, 4, 3, 2, 5))
                # Reshape to (BLOCK_M, SCALES_PER_BLOCK_K)
                a_scale_block = ct.reshape(a_scale_perm, (BLOCK_M, SCALES_PER_BLOCK_K))

                # b_scale is raw [Q, rn, rk, 2, 256] — load block and unswizzle
                # SCALE_REP_N = BLOCK_N // 128 (same structure as SCALE_REP_M)
                # For b_scale, we use pid_n index into rn dimension
                b_scale_5d = ct.load(
                    b_scale,
                    index=(pid_q, pid_n, k, 0, 0),
                    shape=(1, BLOCK_N // 128, SCALE_REP_K, 2, 256),
                    padding_mode=ct.PaddingMode.ZERO,
                )
                b_scale_6d = ct.reshape(
                    b_scale_5d, (1, BLOCK_N // 128, SCALE_REP_K, 32, 4, 4)
                )
                b_scale_perm = ct.permute(b_scale_6d, (0, 1, 4, 3, 2, 5))
                # Reshape to (BLOCK_N, SCALES_PER_BLOCK_K) then transpose to (SCALES_PER_BLOCK_K, BLOCK_N)
                # mma_scaled expects b_scale as (K_scale, N) to match b_block (K, N)
                b_scale_2d = ct.reshape(b_scale_perm, (BLOCK_N, SCALES_PER_BLOCK_K))
                b_scale_block = ct.permute(
                    b_scale_2d, (1, 0)
                )  # -> (SCALES_PER_BLOCK_K, BLOCK_N)

                # Hardware-accelerated scaled MMA
                acc = mma_scaled(a_block, a_scale_block, b_block, b_scale_block, acc)

            # Convert accumulator to output dtype and store
            c_block = ct.astype(acc, c.dtype)
            c_block_3d = ct.reshape(c_block, (1, BLOCK_M, BLOCK_N))
            ct.store(
                c,
                index=(pid_q, pid_m, pid_n),
                tile=c_block_3d,
            )


def _masked_scaled_bmm_autotune_configs(device=None):
    """
    Iterator of autotune configurations for masked_scaled_bmm kernel.

    Returns configurations optimized for different GPU architectures.
    Aligned with Triton's search space for comparable performance.
    """
    gpu_capability = torch.cuda.get_device_capability(device)

    if gpu_capability[0] >= 10:
        # Blackwell family (B200/B300/B100: sm_100/103/120/121)
        # BLOCK_K must be 128 (256 causes TMA misalignment for block-scaled ops).
        # Aligned with Triton configs (BM, BN, nc, occ, gsm):
        for BM, BN, nc, occ, gsm in [
            (128, 256, 1, 2, 1),  # Matches Triton (128, 256, 128, 1, 2)
            (
                128,
                128,
                1,
                4,
                1,
            ),  # Matches Triton (128, 128, 256, 1, 4) — high occupancy
            (128, 128, 1, 1, 1),  # Matches Triton (128, 128, 256, 1, 1)
            (128, 256, 1, 1, 1),  # Matches Triton (128, 256, 256, 1, 1)
            (256, 128, 2, 1, 1),  # Matches Triton (256, 128, 256, 2, 1)
            (256, 256, 2, 1, 1),  # Matches Triton (256, 256, 256, 2, 1)
            (128, 256, 1, 2, 8),  # GROUP_SIZE_M=8 variant
            (128, 128, 1, 4, 8),  # GROUP_SIZE_M=8 variant
        ]:
            yield SimpleNamespace(
                BLOCK_M=BM,
                BLOCK_N=BN,
                BLOCK_K=128,
                GROUP_SIZE_M=gsm,
                num_ctas=nc,
                occupancy=occ,
            )
    elif gpu_capability == (9, 0):
        # H100 (sm_90) - Hopper architecture
        for BM, BN, BK in [(128, 256, 256), (128, 256, 128)]:
            for occupancy in [1, 2]:
                yield SimpleNamespace(
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                    BLOCK_K=BK,
                    GROUP_SIZE_M=8,
                    num_ctas=1,
                    occupancy=occupancy,
                )
    else:
        # Default configurations for other architectures
        for BM, BN, BK in [
            (128, 256, 128),
            (128, 128, 128),
        ]:
            for occupancy in [1, 2]:
                yield SimpleNamespace(
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                    BLOCK_K=BK,
                    GROUP_SIZE_M=8,
                    num_ctas=1,
                    occupancy=occupancy,
                )


def _get_default_kernel_configs(device=None):
    """
    Get GPU-specific default kernel config for the non-autotune path.

    Mirrors the first (fastest on Blackwell) entry of
    _masked_scaled_bmm_autotune_configs. BLOCK_K is fixed to 128 for
    block-scaled ops (256 causes TMA misalignment).
    """
    gpu_capability = torch.cuda.get_device_capability(device)

    if gpu_capability[0] >= 10:
        # Blackwell family (sm_100/103/120/121)
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 1,
            "num_ctas": 1,
            "occupancy": 2,
        }
    elif gpu_capability == (9, 0):
        # H100 (sm_90)
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
        }
    else:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
        }


def _masked_scaled_bmm_autotune_launch(
    stream,
    a,
    b,
    a_scale,
    b_scale,
    masked_m,
    c,
    max_m,
    Q,
    M,
    N,
    K_A,
    K_B,
    ELEM_PER_BYTE_A,
    VEC_SIZE,
    MIXED_PREC,
):
    """
    Autotuned launch for masked_scaled_bmm kernel.

    Runs cuda.tile.tune.exhaustive_search on first call to pick the best
    (BLOCK_M, BLOCK_N, BLOCK_K, occupancy, num_ctas) configuration, caches
    the tuned kernel in _masked_scaled_bmm_tune_cache, and re-launches
    directly on subsequent calls with matching shape/dtype.

    max_m is a 1-element int32 GPU tensor with max(masked_m).
    a_scale, b_scale are raw 5D packed [Q, rm, rk, 2, 256].
    """
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    def args_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        BK = cfg.BLOCK_K
        GSM = cfg.GROUP_SIZE_M

        SCALES_PER_BK = BK // VEC_SIZE
        SCALE_REP_M = BM // 128
        SCALE_REP_K = BK // VEC_SIZE // 4
        # Use M (upper bound) for grid sizing; kernel uses max_m for dynamic tile count

        return (
            a,
            b,
            a_scale,
            b_scale,
            masked_m,
            c,
            max_m,
            Q,
            M,
            N,
            ELEM_PER_BYTE_A,
            VEC_SIZE,
            SCALES_PER_BK,
            MIXED_PREC,
            BM,
            BN,
            BK,
            GSM,
            SCALE_REP_M,
            SCALE_REP_K,
        )

    def grid_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        # Use M (upper bound) for grid sizing
        num_pid_m_upper = ct.cdiv(M, BM)
        num_pid_n = ct.cdiv(N, BN)
        tiles_per_batch_upper = num_pid_m_upper * num_pid_n
        total_tiles_upper = tiles_per_batch_upper * Q
        num_programs = min(NUM_SMS // cfg.num_ctas, total_tiles_upper) * cfg.occupancy
        return (num_programs, 1, 1)

    def hints_fn(cfg):
        return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

    cache_key = (
        Q,
        M,
        N,
        K_A,
        K_B,
        ELEM_PER_BYTE_A,
        VEC_SIZE,
        MIXED_PREC,
        a.dtype,
        str(a.device),
    )
    if cache_key not in _masked_scaled_bmm_tune_cache:
        result = exhaustive_search(
            list(_masked_scaled_bmm_autotune_configs(a.device)),
            stream,
            grid_fn,
            _masked_scaled_bmm_kernel,
            args_fn,
            hints_fn,
        )
        best_cfg = result.best.config
        _masked_scaled_bmm_tune_cache[cache_key] = (
            best_cfg,
            _masked_scaled_bmm_kernel.replace_hints(**hints_fn(best_cfg)),
        )
    best_cfg, tuned_kernel = _masked_scaled_bmm_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


def _masked_scaled_bmm_default_launch(
    stream,
    a,
    b,
    a_scale,
    b_scale,
    masked_m,
    c,
    max_m,
    Q,
    M,
    N,
    ELEM_PER_BYTE_A,
    VEC_SIZE,
    MIXED_PREC,
    kernel_configs,
):
    """
    Non-autotune launch using a fixed default config (respects
    FLASHINFER_CUTILE_AUTOTUNE_DISABLED). Shares the same arg/grid layout as
    the autotune path so the launched kernel is identical.
    """
    BM = kernel_configs["BLOCK_M"]
    BN = kernel_configs["BLOCK_N"]
    BK = kernel_configs["BLOCK_K"]
    GSM = kernel_configs.get("GROUP_SIZE_M", 1)
    num_ctas = kernel_configs.get("num_ctas", 1)
    occupancy = kernel_configs.get("occupancy", 1)

    SCALES_PER_BK = BK // VEC_SIZE
    SCALE_REP_M = BM // 128
    SCALE_REP_K = BK // VEC_SIZE // 4

    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_pid_m_upper = ct.cdiv(M, BM)
    num_pid_n = ct.cdiv(N, BN)
    total_tiles_upper = num_pid_m_upper * num_pid_n * Q
    num_programs = min(NUM_SMS // num_ctas, total_tiles_upper) * occupancy
    grid = (num_programs, 1, 1)

    hints = {}
    if num_ctas is not None:
        hints["num_ctas"] = num_ctas
    if occupancy is not None:
        hints["occupancy"] = occupancy
    kernel = (
        cached_replace_hints(_masked_scaled_bmm_kernel, **hints)
        if hints
        else _masked_scaled_bmm_kernel
    )

    ct.launch(
        stream,
        grid,
        kernel,
        (
            a,
            b,
            a_scale,
            b_scale,
            masked_m,
            c,
            max_m,
            Q,
            M,
            N,
            ELEM_PER_BYTE_A,
            VEC_SIZE,
            SCALES_PER_BK,
            MIXED_PREC,
            BM,
            BN,
            BK,
            GSM,
            SCALE_REP_M,
            SCALE_REP_K,
        ),
    )


def masked_scaled_bmm(
    a,
    b,
    a_scale,
    b_scale,
    masked_m,
    block_scale_type,
    max_m_device=None,
    transpose_a=False,
    transpose_b=True,
    out_dtype=None,
    **kwargs,
):
    """
    cuTile implementation of masked block-scaled batched matrix multiplication.

    Computes C[q] = (A[q] * A_scale[q]) @ (B[q] * B_scale[q])^T where:
    - A is batched FP8/FP4 [Q, M, K_A]
    - B is batched FP8/FP4 [Q, N, K_B]
    - A_scale, B_scale are 5D packed scale tensors [Q, rm, rk, 2, 256]
    - masked_m is [Q] — per-batch valid M count
    - block_scale_type is one of: "mxfp8", "mxfp4", "nvfp4", "mixed"

    Scale unswizzle is done in-kernel (not on host) for performance.

    Args:
        a: Input matrix A (FP8/FP4) [Q, M, K_A]
        b: Input matrix B (FP8/FP4) [Q, N, K_B]
        a_scale: Per-block scale for A [Q, rm, rka, 2, 256]
        b_scale: Per-block scale for B [Q, rn, rkb, 2, 256]
        masked_m: Per-batch valid M count [Q]
        block_scale_type: One of "mxfp8", "mxfp4", "nvfp4", "mixed"
        max_m_device: Optional pre-computed max(masked_m) as a scalar tensor
        transpose_a: Whether A is transposed (must be False)
        transpose_b: Whether B is transposed (must be True)
        out_dtype: Output data type (defaults to bfloat16)

    Returns:
        Output tensor C [Q, M, N]
    """
    # Extract dimensions
    if transpose_a:
        Q_A, K_A, M = a.shape
    else:
        Q_A, M, K_A = a.shape

    if transpose_b:
        Q_B, N, K_B = b.shape
    else:
        Q_B, K_B, N = b.shape

    # Compute ELEM_PER_BYTE based on block_scale_type
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    # Shape sanity checks — use explicit ValueErrors instead of `assert` so the
    # validation isn't elided when Python is run with `-O` (which strips assert
    # statements and would let bad inputs reach the cuda.tile kernel).
    if K_A * ELEM_PER_BYTE_A != K_B * ELEM_PER_BYTE_B:
        raise ValueError(
            f"incompatible dimensions: K_A*ELEM_PER_BYTE_A ({K_A * ELEM_PER_BYTE_A}) "
            f"must match K_B*ELEM_PER_BYTE_B ({K_B * ELEM_PER_BYTE_B})"
        )
    if Q_A != Q_B:
        raise ValueError(f"incompatible dimensions: Q_A ({Q_A}) must match Q_B ({Q_B})")
    Q = Q_A

    if transpose_a or not transpose_b:
        raise ValueError(
            "Only NT layout is supported (transpose_a=False, transpose_b=True)"
        )
    if not a.is_contiguous():
        raise ValueError("A matrix must be contiguous")
    if not b.is_contiguous():
        raise ValueError("B matrix must be contiguous")
    if not a_scale.is_contiguous():
        raise ValueError("A scale matrix must be contiguous")
    if not b_scale.is_contiguous():
        raise ValueError("B scale matrix must be contiguous")
    if not masked_m.is_contiguous():
        raise ValueError("Masked matrix must be contiguous")
    if masked_m.numel() != Q:
        raise ValueError(f"masked_m must have the same number of elements as Q ({Q})")

    if block_scale_type not in ["nvfp4", "mxfp4", "mxfp8", "mixed"]:
        raise ValueError(f"Invalid block scale type: {block_scale_type}")
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32

    # Validate scale shapes
    Q_SA, rm, rka, m2, k256a = a_scale.shape
    Q_SB, rn, rkb, n2, k256b = b_scale.shape
    if m2 != 2 or n2 != 2:
        raise ValueError("incompatible dimensions: scale dim -2 must be 2")
    if k256a != 256 or k256b != 256:
        raise ValueError("incompatible dimensions: scale dim -1 must be 256")
    if Q_SA != Q_SB or Q_SA != Q:
        raise ValueError("incompatible dimensions: scale batch must match Q")
    if rm * 128 != M or rn * 128 != N:
        raise ValueError(
            "incompatible dimensions: M/N must be a multiple of 128 matching scale shape"
        )
    if (
        rka * 4 * VEC_SIZE != K_A * ELEM_PER_BYTE_A
        or rkb * 4 * VEC_SIZE != K_B * ELEM_PER_BYTE_B
    ):
        raise ValueError("incompatible dimensions: K must match scale K shape")

    # Auto-convert scale dtype for mma_scaled.
    # NVFP4: scales are float8_e4m3fn. MX formats: scales are float8_e8m0fnu.
    # Both are byte-identical to uint8, so .view() is a zero-cost operation.
    if block_scale_type == "nvfp4":
        if a_scale.dtype == torch.uint8:
            a_scale = a_scale.view(torch.float8_e4m3fn)
        if b_scale.dtype == torch.uint8:
            b_scale = b_scale.view(torch.float8_e4m3fn)
    else:
        if a_scale.dtype == torch.uint8:
            a_scale = a_scale.view(torch.float8_e8m0fnu)
        if b_scale.dtype == torch.uint8:
            b_scale = b_scale.view(torch.float8_e8m0fnu)

    # Determine output dtype
    if out_dtype is None:
        out_dtype = torch.bfloat16

    # NO host-side unswizzle — scales are passed raw 5D to the kernel.
    # The kernel does in-register reshape/permute/reshape (same as Triton).
    # This avoids extra memory copies that were a major source of overhead.

    # Compute max(masked_m) on GPU — avoids GPU->CPU sync.
    # The kernel uses max_m to dynamically reduce tile scheduling.
    if max_m_device is not None:
        if isinstance(max_m_device, torch.Tensor) and max_m_device.is_cuda:
            max_m = max_m_device.to(torch.int32).reshape(1)
        else:
            # Scalar or CPU tensor — put it on GPU
            val = (
                max_m_device
                if isinstance(max_m_device, int)
                else int(max_m_device.item())
            )
            max_m = torch.tensor([min(val, M)], device=a.device, dtype=torch.int32)
    else:
        max_m = _masked_m_max_device_cutile(masked_m)

    # Derived constants
    MIXED_PREC = 1 if ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2 else 0

    c = torch.empty((Q, M, N), device=a.device, dtype=out_dtype)

    enable_autotune = not _AUTOTUNE_DISABLED

    if enable_autotune:
        # Launch via cuda.tile.tune.exhaustive_search with a module-level tune
        # cache (tuned once per shape/dtype, then replayed).
        _masked_scaled_bmm_autotune_launch(
            torch.cuda.current_stream(a.device),
            a,
            b,
            a_scale,
            b_scale,
            masked_m,
            c,
            max_m,
            Q,
            M,
            N,
            K_A,
            K_B,
            ELEM_PER_BYTE_A,
            VEC_SIZE,
            MIXED_PREC,
        )
    else:
        default_configs = _get_default_kernel_configs(a.device)
        kernel_configs = {**default_configs, **(kwargs.get("kernel_configs") or {})}
        _masked_scaled_bmm_default_launch(
            torch.cuda.current_stream(a.device),
            a,
            b,
            a_scale,
            b_scale,
            masked_m,
            c,
            max_m,
            Q,
            M,
            N,
            ELEM_PER_BYTE_A,
            VEC_SIZE,
            MIXED_PREC,
            kernel_configs,
        )
    return c
