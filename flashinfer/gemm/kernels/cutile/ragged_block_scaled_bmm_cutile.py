# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT


import cuda.tile as ct
import torch

from ....cutile.cutile_common import cached_replace_hints

# The kernel addresses each segment's rows via a tile index `m_start // BLOCK_M`
# (it cannot use Array.slice(): cuTile's dynamic-TMA slice path IMAs on
# runtime-computed offsets). That division is only exact when every segment offset
# is a multiple of the *selected* BLOCK_M; otherwise it truncates and the kernel
# silently reads the wrong rows -> corrupt output. A runtime alignment check would
# need a host sync that breaks this kernel's CUDA-graph capturability, so instead
# the caller declares the alignment it guarantees via `segment_alignment` and the
# host only selects BLOCK_M values that divide it. Default 128 matches the minimum
# cuTile segment alignment; pass 256 (with 256-aligned segments) to re-enable the
# large-M BLOCK_M=256 fast path.
_DEFAULT_SEGMENT_ALIGNMENT = 128


def _is_large_m(total_m, Q):
    """Determine if average M is large enough for non-swapped configs."""
    average_m = total_m / Q
    is_large_m = average_m >= 256
    return is_large_m


@ct.kernel
def _ragged_block_scaled_bmm_kernel(
    a,  # Input matrix A [total_m, K] FP8
    b,  # Input matrix B [Q, N, K] FP8
    a_scale,  # Scale for A [total_m, k_tiles] FP32
    b_scale,  # Scale for B [Q, n_tiles, k_tiles] FP32
    c,  # Output matrix C [total_m, N]
    m_indptr,  # Segment offsets [Q+1], flattened 1D
    q,  # Number of batches
    max_m,  # Host-side max segment size hint (kept for autotune cache key)
    max_m_device,  # 1-element int32 tensor (shape (1,)) — device-side ground truth for max(valid_m)
    n,  # Output N dimension
    HAS_A_SCALE: ct.Constant[int],  # Whether a_scale is provided (0 or 1)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    cuTile kernel for ragged block-scaled batched matrix multiplication.

    Performs (A * a_scale) @ (B * b_scale)^T where:
    - A is flattened FP8 with segment offsets (m_indptr defines boundaries)
    - B is batched FP8 [Q, N, K]
    - a_scale and b_scale are per-block scales
    - Output C is [total_m, N]

    Uses persistent scheduling with static grid and GROUP_SIZE_M tile swizzling.
    Uses Array.slice + TMA (ct.load/ct.store) for A and C access.

    Defense-in-depth: the per-tile loop bound is computed from the device-side
    `max_m_device` rather than the host-side `max_m`. This prevents silent output corruption when the caller passes
    too small a host-side max_m hint.
    This matches the NVT triton kernel's defensive semantic.
    """
    pid = ct.bid(0)

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
        # pid_q = batch index
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
            # Compute tile-level offset into the flattened global A/C tensors.
            # This integer division is exact only when m_start is BLOCK_M-aligned;
            # the host only selects a BLOCK_M that divides the caller-declared
            # segment_alignment (see _DEFAULT_SEGMENT_ALIGNMENT) so truncation cannot
            # occur. We avoid Array.slice() because cuTile's dynamic TMA descriptor
            # path for slice has a bug that IMAs on runtime-computed offsets.
            m_tile_start = m_start // BLOCK_M

            # Initialize accumulator
            acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            # N tile offset (element-level) for b_scale calculation
            n_offset = pid_n * BLOCK_N
            offs_bsn = n_offset // BLOCK_K

            # Zero accumulator for per-K MMA (reused each iteration)
            mma_zeros = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            # K-loop for matrix multiplication
            for k in range(num_k_tiles):
                k_offset = k * BLOCK_K

                # Load A block using TMA (direct global index, no slice)
                a_block = ct.load(
                    a,
                    index=(m_tile_start + pid_m, k),
                    shape=(BLOCK_M, BLOCK_K),
                    padding_mode=ct.PaddingMode.ZERO,
                )

                # Load B block - B is [Q, N, K], we need [BLOCK_N, BLOCK_K]
                b_block_3d = ct.load(
                    b,
                    index=(pid_q, n_offset // BLOCK_N, k_offset // BLOCK_K),
                    shape=(1, BLOCK_N, BLOCK_K),
                    order=(0, 1, 2),
                    padding_mode=ct.PaddingMode.ZERO,
                )
                # Reshape to [BLOCK_N, BLOCK_K] then transpose to get [BLOCK_K, BLOCK_N]
                b_block_nk = ct.reshape(b_block_3d, (BLOCK_N, BLOCK_K))
                b_block = ct.permute(b_block_nk, (1, 0))  # [BLOCK_K, BLOCK_N]

                # Matrix multiplication: A [BLOCK_M, BLOCK_K] @ B [BLOCK_K, BLOCK_N] = C [BLOCK_M, BLOCK_N]
                c_mma = ct.mma(a_block, b_block, acc=mma_zeros)

                # Load and apply scales
                if HAS_A_SCALE == 1:
                    # Load a_scale for this block using TMA (direct global index)
                    a_scale_block = ct.load(
                        a_scale,
                        index=(m_tile_start + pid_m, k),
                        shape=(BLOCK_M, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                    )

                    # Load b_scale - scalar at [pid_q, offs_bsn, k]
                    b_scale_block = ct.load(
                        b_scale,
                        index=(pid_q, offs_bsn, k),
                        shape=(1, 1, 1),
                        order=(0, 1, 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_scale_val = ct.reshape(b_scale_block, (1, 1))

                    # Combined scale: a_scale [BLOCK_M, 1] * b_scale [1, 1] = [BLOCK_M, 1]
                    scale_combined = a_scale_block * ct.broadcast_to(
                        b_scale_val, (BLOCK_M, 1)
                    )
                    scale_ab = ct.broadcast_to(scale_combined, (BLOCK_M, BLOCK_N))
                else:
                    # Only b_scale
                    b_scale_block = ct.load(
                        b_scale,
                        index=(pid_q, offs_bsn, k),
                        shape=(1, 1, 1),
                        order=(0, 1, 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_scale_val = ct.reshape(b_scale_block, (1, 1))
                    scale_ab = ct.broadcast_to(b_scale_val, (BLOCK_M, BLOCK_N))

                # Apply scale and accumulate
                acc = acc + c_mma * scale_ab

            # Convert to output dtype
            c_block = ct.astype(acc, c.dtype)

            # Store to output C using TMA (direct global index, no slice)
            ct.store(c, index=(m_tile_start + pid_m, pid_n), tile=c_block)


def _get_default_kernel_configs(
    total_m, Q, VEC_SIZE, segment_alignment=_DEFAULT_SEGMENT_ALIGNMENT
):
    """
    Get GPU-specific default kernel configs for non-autotune path.

    ``segment_alignment`` bounds the largest usable BLOCK_M: the sm90 large-M
    fast path (BLOCK_M=256) is only selected when the caller guarantees
    256-aligned segments, otherwise it falls back to the 128 config.
    """
    gpu_capability = torch.cuda.get_device_capability()
    is_large_m = _is_large_m(total_m, Q)

    if gpu_capability in [(12, 0), (12, 1)]:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": VEC_SIZE,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 2,
        }
    elif gpu_capability == (9, 0):
        # Large-M prefers BLOCK_M=256, but m_start // BLOCK_M requires every
        # segment offset to be 256-aligned. Only take it when the caller
        # guarantees that via segment_alignment; else fall back to 128.
        if is_large_m and segment_alignment % 256 == 0:
            return {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": VEC_SIZE,
                "GROUP_SIZE_M": 8,
                "num_ctas": 2,
                "occupancy": 1,
            }
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": VEC_SIZE,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
        }
    else:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": VEC_SIZE,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
        }


def ragged_block_scaled_bmm(
    a,
    b,
    a_scale,
    b_scale,
    m_indptr,
    max_m,
    max_m_device=None,
    transpose_a=False,
    transpose_b=True,
    out_dtype=None,
    out=None,
    segment_alignment=_DEFAULT_SEGMENT_ALIGNMENT,
    **kwargs,
):
    """
    cuTile implementation of ragged block-scaled BMM.

    `max_m_device` is an optional [1]-shape int tensor with the device-side
    ground truth for max(per-batch valid_m). When provided, the kernel uses it
    for its persistent-loop bound — preventing silent corruption if the host-side `max_m` hint
    underestimates the actual per-batch max. When None, a fallback tensor is
    materialized from `max_m`.
    This mirrors the NVT triton kernel's defensive semantic.

    `segment_alignment` is the alignment (in rows) the caller guarantees for every
    `m_indptr` segment offset; it bounds the largest BLOCK_M the kernel may select
    (BLOCK_M must divide it) because the kernel indexes rows as `m_start // BLOCK_M`.
    Default 128. Pass 256 (with 256-aligned segments) to enable the large-M
    BLOCK_M=256 fast path. This cannot be validated at runtime without a host sync
    that would break CUDA-graph capture, so it is a caller contract.

    `out` is an optional pre-allocated ``(total_m, N)`` output tensor written in
    place (must be contiguous and match the output dtype). When None, an output is
    allocated and returned. Passing `out` lets callers avoid an extra copy.
    """
    # Validate inputs
    assert not transpose_a and transpose_b, "Only NT layout is supported"
    assert a.is_contiguous(), "A matrix must be contiguous"
    assert b.is_contiguous(), "B matrix must be contiguous"
    assert a_scale is None or a_scale.is_contiguous(), (
        "A scale matrix must be contiguous"
    )
    assert b_scale.is_contiguous(), "B scale matrix must be contiguous"
    assert m_indptr.is_contiguous(), "m_indptr must be contiguous"

    # Get dimensions
    total_m, K_A = a.shape
    Q, N, K_B = b.shape

    assert K_A == K_B, f"K dimensions must match: {K_A} != {K_B}"
    assert m_indptr.shape[0] == Q + 1, "m_indptr must have Q+1 elements"

    # Validate scale dimensions
    Q_SB, rnb, rkb = b_scale.shape
    VEC_SIZE = K_B // rkb

    if a_scale is not None:
        total_ma, rka = a_scale.shape
        assert total_ma == total_m, "a_scale total_m dimension mismatch"

    assert Q_SB == Q, "b_scale Q dimension mismatch"

    # Determine output dtype
    if out_dtype is None:
        out_dtype = out.dtype if out is not None else torch.bfloat16

    # Output tensor. The kernel stores directly into it in the output dtype (no
    # float32 intermediate). A caller-provided ``out`` is written in place (lets
    # callers like group_gemm_fp8_nt_groupwise avoid an extra output copy);
    # otherwise it is allocated here.
    if out is not None:
        if tuple(out.shape) != (total_m, N):
            raise ValueError(f"out must be ({total_m}, {N}); got {tuple(out.shape)}")
        if out.dtype != out_dtype:
            raise ValueError(f"out.dtype {out.dtype} != out_dtype {out_dtype}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        c = out
    else:
        c = torch.empty((total_m, N), device=a.device, dtype=out_dtype)

    # Materialize fallback max_m_device if the caller didn't pass one. The
    # kernel always reads its grid bound from a device tensor (defense-in-depth).
    if max_m_device is None:
        max_m_device = torch.tensor([max_m], dtype=torch.int32, device=a.device)

    # Get kernel configs
    default_configs = _get_default_kernel_configs(
        total_m, Q, VEC_SIZE, segment_alignment
    )
    kernel_configs = {**default_configs, **(kwargs.get("kernel_configs") or {})}

    BLOCK_M = kernel_configs.get("BLOCK_M")
    BLOCK_N = kernel_configs.get("BLOCK_N")
    BLOCK_K = kernel_configs.get("BLOCK_K", VEC_SIZE)
    GROUP_SIZE_M = kernel_configs.get("GROUP_SIZE_M", 8)
    num_ctas = kernel_configs.get("num_ctas", 1)
    occupancy = kernel_configs.get("occupancy", 1)

    # The kernel indexes segment rows as `m_start // BLOCK_M`, so every segment
    # offset must be a multiple of BLOCK_M. Guard here (also catches an explicit
    # kernel_configs BLOCK_M override) rather than silently corrupting output.
    if segment_alignment % BLOCK_M != 0:
        raise ValueError(
            f"BLOCK_M ({BLOCK_M}) must divide segment_alignment ({segment_alignment}); "
            "align m_indptr segments to a multiple of BLOCK_M or pass a smaller BLOCK_M."
        )

    # Calculate grid size for persistent scheduling. Use total_m (a guaranteed
    # host-side upper bound, always > 0 for nonempty input) rather than the
    # max_m hint: a stale/zero max_m would make num_programs 0 and leave the
    # output unwritten. The kernel still reads its per-tile bound from
    # max_m_device, so this only affects the launched program count.
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_pid_m = ct.cdiv(total_m, BLOCK_M)
    num_pid_n = ct.cdiv(N, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * Q
    num_programs = min(NUM_SMS // num_ctas, total_tiles) * occupancy

    grid = (num_programs, 1, 1)

    has_a_scale = 1 if a_scale is not None else 0

    if a_scale is None:
        a_scale = torch.empty(1, device=a.device, dtype=torch.float32)
    else:
        a_scale = a_scale

    kernel_fn = _ragged_block_scaled_bmm_kernel

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
            a_scale,
            b_scale,
            c,
            m_indptr,
            Q,
            max_m,
            max_m_device,
            N,
            has_a_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_SIZE_M,
        ),
    )

    return c
