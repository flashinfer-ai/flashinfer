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

# Module-level tune cache: (Q, max_m, N, K_A, K_B, ELEM_PER_BYTE_A,
# ELEM_PER_BYTE_B, VEC_SIZE, MIXED_PREC, has_a_gs, has_b_gs, dtype, device)
# -> (best_cfg, tuned_kernel)
_ragged_scaled_bmm_tune_cache: dict = {}


def _ragged_scaled_bmm_autotune_configs(device=None):
    """
    Iterator of autotune configurations for ragged_scaled_bmm kernel.
    """
    gpu_capability = torch.cuda.get_device_capability(device)

    if gpu_capability[0] >= 10:
        # Blackwell family
        for BM, BN, nc, occ, gsm in [
            (128, 128, 1, 2, 8),
            (128, 128, 1, 1, 8),
            (128, 256, 1, 1, 8),
            (128, 256, 1, 2, 8),
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
        # Hopper
        for BM, BN, BK in [(128, 128, 128)]:
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
        # Ampere/Ada
        for BM, BN, BK in [(128, 128, 128)]:
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

    Mirrors the first entry of _ragged_scaled_bmm_autotune_configs. BLOCK_K is
    fixed to 128 for block-scaled ops (256 causes TMA misalignment).
    """
    gpu_capability = torch.cuda.get_device_capability(device)

    if gpu_capability[0] >= 10:
        # Blackwell family (sm_100/103/120/121)
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 2,
        }
    else:
        # Hopper / Ampere / Ada
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 128,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
        }


@ct.kernel
def _ragged_scaled_bmm_kernel(
    a,
    b,
    a_scale,
    b_scale,
    a_global_scale_tensor,
    b_global_scale_tensor,
    m_indptr,
    c,
    q,
    max_m,
    n,
    ELEM_PER_BYTE_A: ct.Constant[int],
    ELEM_PER_BYTE_B: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    has_a_global_scale: ct.Constant[int],
    has_b_global_scale: ct.Constant[int],
    SCALES_PER_BLOCK_K: ct.Constant[int],
    SCALE_REP_M: ct.Constant[int],
    SCALE_REP_K: ct.Constant[int],
    MIXED_PREC: ct.Constant[int],
):
    pid = ct.bid(0)
    if ELEM_PER_BYTE_A == 2:
        num_k_tiles = ct.num_tiles(a, axis=1, shape=(BLOCK_M, BLOCK_K // 2))
    else:
        num_k_tiles = ct.num_tiles(a, axis=1, shape=(BLOCK_M, BLOCK_K))
    num_pid_m = ct.cdiv(max_m, BLOCK_M)
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

        # Load segment boundaries
        m_start = ct.load(m_indptr, index=(pid_q,), shape=(1,)).item()
        m_end = ct.load(m_indptr, index=(pid_q + 1,), shape=(1,)).item()
        valid_m = m_end - m_start

        if pid_m * BLOCK_M < valid_m:
            # Create sliced views for A, C and swizzled A-scale.
            # Host side asserts swizzled_layout_a=True, so a_scale is
            # [total_m // 128, rka, 2, 256] and slicing by m_start//128 is valid.
            Ai = a.slice(axis=0, start=m_start, stop=m_end)
            Ci = c.slice(axis=0, start=m_start, stop=m_end)
            a_scale_i = a_scale.slice(axis=0, start=m_start // 128, stop=m_end // 128)

            # Initialize accumulator
            acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            for k in range(num_k_tiles):
                # Load A
                if ELEM_PER_BYTE_A == 2:  # FP4
                    # Load [BLOCK_M, BLOCK_K//2] -> unpack -> [BLOCK_M, BLOCK_K]
                    a_bytes = ct.load(
                        Ai,
                        index=(pid_m, k),
                        shape=(BLOCK_M, BLOCK_K // 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    a_flat = ct.reshape(a_bytes, (-1,))
                    a_unpacked = ct.unpack_from_bytes(a_flat, ct.float4_e2m1fn)
                    a_block = ct.reshape(a_unpacked, (BLOCK_M, BLOCK_K))
                else:  # FP8
                    a_block = ct.load(
                        Ai,
                        index=(pid_m, k),
                        shape=(BLOCK_M, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )

                # Load B [Q, N, K_B] -> [BLOCK_K, BLOCK_N]
                if ELEM_PER_BYTE_B == 2:  # FP4
                    # Load [1, BLOCK_N, BLOCK_K//2] -> unpack -> [BLOCK_N, BLOCK_K] -> permute
                    b_bytes = ct.load(
                        b,
                        index=(pid_q, pid_n, k),
                        shape=(1, BLOCK_N, BLOCK_K // 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_flat = ct.reshape(b_bytes, (-1,))
                    b_unpacked = ct.unpack_from_bytes(b_flat, ct.float4_e2m1fn)
                    b_nk = ct.reshape(b_unpacked, (BLOCK_N, BLOCK_K))
                    b_block = ct.permute(b_nk, (1, 0))
                else:  # FP8
                    b_nk = ct.load(
                        b,
                        index=(pid_q, pid_n, k),
                        shape=(1, BLOCK_N, BLOCK_K),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_nk = ct.reshape(b_nk, (BLOCK_N, BLOCK_K))
                    b_block = ct.permute(b_nk, (1, 0))

                # Mixed Precision Handling
                # mma_scaled requires both operands to have the same dtype.
                # Triton's tl.dot_scaled supports cross-precision natively, but
                # mma_scaled does not. We downcast A from fp8 to fp4 as a
                # workaround, which may lose precision vs Triton.
                if MIXED_PREC == 1:
                    a_block = ct.astype(a_block, ct.float4_e2m1fn)

                # A Scale (Swizzled 4D on sliced array)
                # a_scale_i shape after slice: [valid_m//128, K//VEC_SIZE//4, 2, 256]
                # Load shape: (SCALE_REP_M, SCALE_REP_K, 2, 256)
                a_scale_4d = ct.load(
                    a_scale_i,
                    index=(pid_m, k, 0, 0),
                    shape=(SCALE_REP_M, SCALE_REP_K, 2, 256),
                    padding_mode=ct.PaddingMode.ZERO,
                )
                # Unswizzle: (REP_M, REP_K, 32, 4, 4) -> permute(0, 3, 2, 1, 4) -> (BM, BK_SCALES)
                a_scale_5d = ct.reshape(
                    a_scale_4d, (SCALE_REP_M, SCALE_REP_K, 32, 4, 4)
                )
                a_scale_perm = ct.permute(a_scale_5d, (0, 3, 2, 1, 4))
                a_scale_block = ct.reshape(a_scale_perm, (BLOCK_M, SCALES_PER_BLOCK_K))

                # B Scale (Swizzled 5D)
                # b_scale shape: [Q, N//128, K//VEC_SIZE//4, 2, 256]
                # Load shape: (1, BLOCK_N//128, SCALE_REP_K, 2, 256)
                b_scale_5d = ct.load(
                    b_scale,
                    index=(pid_q, pid_n, k, 0, 0),
                    shape=(1, BLOCK_N // 128, SCALE_REP_K, 2, 256),
                    padding_mode=ct.PaddingMode.ZERO,
                )
                # Unswizzle: (1, REP_N, REP_K, 32, 4, 4) -> permute(0, 1, 4, 3, 2, 5) -> (BN, BK_SCALES) -> permute(1, 0)
                b_scale_6d = ct.reshape(
                    b_scale_5d, (1, BLOCK_N // 128, SCALE_REP_K, 32, 4, 4)
                )
                b_scale_perm = ct.permute(b_scale_6d, (0, 1, 4, 3, 2, 5))
                b_scale_2d = ct.reshape(b_scale_perm, (BLOCK_N, SCALES_PER_BLOCK_K))
                b_scale_block = ct.permute(b_scale_2d, (1, 0))

                # MMA Scaled
                acc = mma_scaled(a_block, a_scale_block, b_block, b_scale_block, acc)

            # Apply Global Scales
            a_gs = 1.0
            if has_a_global_scale == 1:
                a_gs = ct.load(a_global_scale_tensor, index=(0,), shape=(1,)).item()

            b_gs = 1.0
            if has_b_global_scale == 1:
                # b_global_scale is per-batch [Q]
                b_gs = ct.load(b_global_scale_tensor, index=(pid_q,), shape=(1,)).item()

            global_scale_val = 1.0 / (a_gs * b_gs)
            acc = acc * global_scale_val

            # Store C
            c_block = ct.astype(acc, c.dtype)
            ct.store(Ci, index=(pid_m, pid_n), tile=c_block)


def _ragged_scaled_bmm_launch(
    stream,
    a,
    b,
    a_scale,
    b_scale,
    a_global_scale_tensor,
    b_global_scale_tensor,
    m_indptr,
    c,
    Q,
    max_m,
    N,
    K_A,
    K_B,
    ELEM_PER_BYTE_A,
    ELEM_PER_BYTE_B,
    VEC_SIZE,
    MIXED_PREC,
    has_a_global_scale,
    has_b_global_scale,
):
    """
    Autotuned launch for the ragged_scaled_bmm kernel.

    Runs cuda.tile.tune.exhaustive_search on first call to pick the best config,
    caches the tuned kernel in _ragged_scaled_bmm_tune_cache, and re-launches
    directly on subsequent calls with matching shape/dtype.
    """
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    def args_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        BK = cfg.BLOCK_K
        GSM = cfg.GROUP_SIZE_M

        SCALES_PER_BLOCK_K = BK // VEC_SIZE
        SCALE_REP_M = BM // 128
        SCALE_REP_K = BK // VEC_SIZE // 4

        return (
            a,
            b,
            a_scale,
            b_scale,
            a_global_scale_tensor,
            b_global_scale_tensor,
            m_indptr,
            c,
            Q,
            max_m,
            N,
            ELEM_PER_BYTE_A,
            ELEM_PER_BYTE_B,
            BM,
            BN,
            BK,
            GSM,
            has_a_global_scale,
            has_b_global_scale,
            SCALES_PER_BLOCK_K,
            SCALE_REP_M,
            SCALE_REP_K,
            MIXED_PREC,
        )

    def grid_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        num_pid_m = ct.cdiv(max_m, BM)
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
        K_A,
        K_B,
        ELEM_PER_BYTE_A,
        ELEM_PER_BYTE_B,
        VEC_SIZE,
        MIXED_PREC,
        has_a_global_scale,
        has_b_global_scale,
        a.dtype,
        str(a.device),
    )
    if cache_key not in _ragged_scaled_bmm_tune_cache:
        result = exhaustive_search(
            list(_ragged_scaled_bmm_autotune_configs(a.device)),
            stream,
            grid_fn,
            _ragged_scaled_bmm_kernel,
            args_fn,
            hints_fn,
        )
        best_cfg = result.best.config
        _ragged_scaled_bmm_tune_cache[cache_key] = (
            best_cfg,
            _ragged_scaled_bmm_kernel.replace_hints(**hints_fn(best_cfg)),
        )
    best_cfg, tuned_kernel = _ragged_scaled_bmm_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


def _ragged_scaled_bmm_default_launch(
    stream,
    a,
    b,
    a_scale,
    b_scale,
    a_global_scale_tensor,
    b_global_scale_tensor,
    m_indptr,
    c,
    Q,
    max_m,
    N,
    ELEM_PER_BYTE_A,
    ELEM_PER_BYTE_B,
    VEC_SIZE,
    MIXED_PREC,
    has_a_global_scale,
    has_b_global_scale,
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
    GSM = kernel_configs.get("GROUP_SIZE_M", 8)
    num_ctas = kernel_configs.get("num_ctas", 1)
    occupancy = kernel_configs.get("occupancy", 1)

    SCALES_PER_BLOCK_K = BK // VEC_SIZE
    SCALE_REP_M = BM // 128
    SCALE_REP_K = BK // VEC_SIZE // 4

    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_pid_m = ct.cdiv(max_m, BM)
    num_pid_n = ct.cdiv(N, BN)
    total_tiles = num_pid_m * num_pid_n * Q
    num_programs = min(NUM_SMS // num_ctas, total_tiles) * occupancy
    grid = (num_programs, 1, 1)

    hints = {}
    if num_ctas is not None:
        hints["num_ctas"] = num_ctas
    if occupancy is not None:
        hints["occupancy"] = occupancy
    kernel = (
        cached_replace_hints(_ragged_scaled_bmm_kernel, **hints)
        if hints
        else _ragged_scaled_bmm_kernel
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
            a_global_scale_tensor,
            b_global_scale_tensor,
            m_indptr,
            c,
            Q,
            max_m,
            N,
            ELEM_PER_BYTE_A,
            ELEM_PER_BYTE_B,
            BM,
            BN,
            BK,
            GSM,
            has_a_global_scale,
            has_b_global_scale,
            SCALES_PER_BLOCK_K,
            SCALE_REP_M,
            SCALE_REP_K,
            MIXED_PREC,
        ),
    )


def ragged_scaled_bmm(
    a,
    b,
    a_scale,
    b_scale,
    m_indptr,
    max_m,
    block_scale_type,
    transpose_a=False,
    transpose_b=True,
    static_persistent=True,
    swizzled_layout_a=True,
    a_global_scale=None,
    b_global_scale=None,
    **kwargs,
):
    """
    cuTile implementation of ragged block-scaled batched matrix multiplication.

    Computes, per expert/segment q, C[seg] = (A[seg] * A_scale) @ (B[q] * B_scale)^T
    where A is a ragged stack [total_m, K_A] partitioned by m_indptr [Q+1], B is
    batched FP8/FP4 [Q, N, K_B], and A_scale/B_scale are MX-swizzled scale
    tensors. Output C is [total_m, N] (float32).

    Args:
        a: Input matrix A (FP8/FP4) [total_m, K_A]
        b: Input matrix B (FP8/FP4) [Q, N, K_B]
        a_scale: Swizzled A scale [total_m // 128, rka, 2, 256]
        b_scale: Swizzled B scale [Q, N // 128, rkb, 2, 256]
        m_indptr: Segment offsets [Q+1] (each entry must be a multiple of 128)
        max_m: Upper bound on any single segment length (used for grid sizing)
        block_scale_type: One of "mxfp8", "mxfp4", "nvfp4", "mixed"
        transpose_a: Whether A is transposed (must be False)
        transpose_b: Whether B is transposed (must be True)
        static_persistent: Unused, kept for API compatibility
        swizzled_layout_a: Whether A-scale uses the swizzled layout (must be True)
        a_global_scale: Optional scalar global scale for A
        b_global_scale: Optional per-batch [Q] global scale for B

    Returns:
        Output tensor C [total_m, N] (float32)
    """
    total_m, K_A = a.shape
    Q, N, K_B = b.shape

    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    # Shape sanity checks — use explicit ValueErrors instead of `assert` so the
    # validation isn't elided when Python is run with `-O` (which strips assert
    # statements and would let bad inputs reach the cuda.tile kernel).
    if block_scale_type not in ["nvfp4", "mxfp4", "mxfp8", "mixed"]:
        raise ValueError(f"Invalid block scale type: {block_scale_type}")
    if K_A * ELEM_PER_BYTE_A != K_B * ELEM_PER_BYTE_B:
        raise ValueError(
            f"incompatible dimensions: K_A*ELEM_PER_BYTE_A ({K_A * ELEM_PER_BYTE_A}) "
            f"must match K_B*ELEM_PER_BYTE_B ({K_B * ELEM_PER_BYTE_B})"
        )
    if transpose_a or not transpose_b:
        raise ValueError(
            "Only NT layout is supported (transpose_a=False, transpose_b=True)"
        )
    # cuTile currently only implements the swizzled A-scale path. The unswizzled
    # path would need per-batch M masking (Triton uses mask=offs_asm < m_end) to
    # avoid reading across segment boundaries; not yet implemented.
    #
    # Alignment requirement (enforced by the caller, mirrors the Triton impl):
    # every entry of m_indptr must be a multiple of 128 so that the kernel's
    # `a_scale.slice(axis=0, start=m_start // 128, stop=m_end // 128)`
    # slicing does not drop the last partial scale row. FlashInfer MoE
    # routing (moe_align_block_size) produces 128-aligned indptrs; we do not
    # validate this on host because m_indptr is on-device and a host-side
    # check would force a GPU->CPU sync.
    if not swizzled_layout_a:
        raise ValueError(
            "Only swizzled_layout_a=True is supported by the cuTile backend"
        )
    if not a.is_contiguous():
        raise ValueError("A matrix must be contiguous")
    if not b.is_contiguous():
        raise ValueError("B matrix must be contiguous")
    if not a_scale.is_contiguous():
        raise ValueError("A scale matrix must be contiguous")
    if not b_scale.is_contiguous():
        raise ValueError("B scale matrix must be contiguous")
    if not m_indptr.is_contiguous():
        raise ValueError("m_indptr must be contiguous")
    if m_indptr.numel() != Q + 1:
        raise ValueError(f"m_indptr must have Q+1 ({Q + 1}) elements")

    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    MIXED_PREC = 1 if ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2 else 0

    # Validate scale shapes (parallels Triton impl).
    Q_SB, rnb, rkb, n2, k256b = b_scale.shape
    if Q_SB != Q:
        raise ValueError("incompatible b_scale batch dim")
    if n2 != 2 or k256b != 256:
        raise ValueError("incompatible b_scale inner dims")
    if rnb * 128 != N:
        raise ValueError("incompatible b_scale N dim")
    if rkb * 4 * VEC_SIZE != K_B * ELEM_PER_BYTE_B:
        raise ValueError("incompatible b_scale K dim")
    total_ma, rka, m2, k256a = a_scale.shape
    if total_ma * 128 != total_m:
        raise ValueError("incompatible a_scale total_m dim")
    if m2 != 2 or k256a != 256:
        raise ValueError("incompatible a_scale inner dims")
    if rka * 4 * VEC_SIZE != K_A * ELEM_PER_BYTE_A:
        raise ValueError("incompatible a_scale K dim")

    # Convert scale dtypes for mma_scaled.
    # For NVFP4 (FP4 E2M1 data, VEC_SIZE=16): mma_scaled accepts f8e4m3fn scales directly.
    # For MXFP8 (FP8 data, VEC_SIZE=32): scales are uint8 representing e8m0fnu, need .view().
    if block_scale_type == "nvfp4":
        # NVFP4: scales are float8_e4m3fn — pass through unchanged.
        # mma_scaled supports f4e2m1fn inputs with f8e4m3fn scales (V=16).
        if a_scale.dtype == torch.uint8:
            a_scale = a_scale.view(torch.float8_e4m3fn)
        if b_scale.dtype == torch.uint8:
            b_scale = b_scale.view(torch.float8_e4m3fn)
    else:
        # MXFP8/MXFP4/mixed: scales are uint8 representing e8m0fnu.
        if a_scale.dtype == torch.uint8:
            a_scale = a_scale.view(torch.float8_e8m0fnu)
        elif a_scale.dtype == torch.float8_e4m3fn:
            a_scale = a_scale.view(torch.uint8).view(torch.float8_e8m0fnu)
        if b_scale.dtype == torch.uint8:
            b_scale = b_scale.view(torch.float8_e8m0fnu)
        elif b_scale.dtype == torch.float8_e4m3fn:
            b_scale = b_scale.view(torch.uint8).view(torch.float8_e8m0fnu)

    c = torch.empty((total_m, N), device=a.device, dtype=torch.float32)

    # Handle optional global scales
    has_a_global_scale = 1 if a_global_scale is not None else 0
    has_b_global_scale = 1 if b_global_scale is not None else 0

    if a_global_scale is None:
        a_global_scale_tensor = torch.empty(1, device=a.device, dtype=torch.float32)
    else:
        a_global_scale_tensor = (
            a_global_scale.reshape(1) if a_global_scale.ndim == 0 else a_global_scale
        )

    if b_global_scale is None:
        b_global_scale_tensor = torch.empty(1, device=a.device, dtype=torch.float32)
    else:
        b_global_scale_tensor = (
            b_global_scale.reshape(1) if b_global_scale.ndim == 0 else b_global_scale
        )

    enable_autotune = not _AUTOTUNE_DISABLED

    if enable_autotune:
        # Launch via cuda.tile.tune.exhaustive_search with a module-level tune
        # cache (tuned once per shape/dtype, then replayed).
        _ragged_scaled_bmm_launch(
            torch.cuda.current_stream(a.device),
            a,
            b,
            a_scale,
            b_scale,
            a_global_scale_tensor,
            b_global_scale_tensor,
            m_indptr,
            c,
            Q,
            max_m,
            N,
            K_A,
            K_B,
            ELEM_PER_BYTE_A,
            ELEM_PER_BYTE_B,
            VEC_SIZE,
            MIXED_PREC,
            has_a_global_scale,
            has_b_global_scale,
        )
    else:
        default_configs = _get_default_kernel_configs(a.device)
        kernel_configs = {**default_configs, **(kwargs.get("kernel_configs") or {})}
        _ragged_scaled_bmm_default_launch(
            torch.cuda.current_stream(a.device),
            a,
            b,
            a_scale,
            b_scale,
            a_global_scale_tensor,
            b_global_scale_tensor,
            m_indptr,
            c,
            Q,
            max_m,
            N,
            ELEM_PER_BYTE_A,
            ELEM_PER_BYTE_B,
            VEC_SIZE,
            MIXED_PREC,
            has_a_global_scale,
            has_b_global_scale,
            kernel_configs,
        )

    return c
