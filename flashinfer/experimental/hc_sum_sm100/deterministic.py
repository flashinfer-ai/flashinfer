# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Derived from sgl-project/sglang at 9cdd6624fe115979b0a0e6c588d9de5ddfeaafe2.
# Modified 2026-11-04: standalone FlashInfer backend; remove framework dependencies.
# Original batch-invariant implementation: Copyright 2025 Thinking Machines Lab.
# MIT text: licenses/LICENSE.batch-invariant-ops.txt.
# https://github.com/thinking-machines-lab/batch_invariant_ops

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def bmm_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    B,
    M,
    N,
    K,  #
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """
    Batched matrix multiplication kernel that processes batches in parallel.
    Each tile processes a (BLOCK_SIZE_M, BLOCK_SIZE_N) output block for a specific batch.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles_per_batch = num_pid_m * num_pid_n
    num_tiles_total = B * num_tiles_per_batch

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Process tiles in a deterministic order: batch-major ordering
    for tile_id in tl.range(start_pid, num_tiles_total, NUM_SMS, flatten=True):
        # Decompose tile_id into batch and within-batch tile
        batch_idx = tile_id // num_tiles_per_batch
        tile_in_batch = tile_id % num_tiles_per_batch

        pid_m, pid_n = _compute_pid(
            tile_in_batch, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Add batch offset
        # A small input can still produce >2**31 output elements. Widen the
        # batch product BEFORE adding it to the already-wide output row offset.
        if A_LARGE or B_LARGE or C_LARGE:
            batch_idx_typed = batch_idx.to(tl.int64)
        else:
            batch_idx_typed = batch_idx

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

            a_ptrs = a_ptr + (
                batch_idx_typed * stride_ab
                + offs_am[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                batch_idx_typed * stride_bb
                + offs_k[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = (
            c_ptr
            + batch_idx_typed * stride_cb
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        elif c_ptr.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif c_ptr.dtype.element_ty == tl.float32:
            c = accumulator.to(tl.float32)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def bmm(a, b, out):
    """Fixed-K-order BMM used by the HC deterministic mode (fixed T only)."""
    batch, m, k = a.shape
    n = b.shape[-1]
    sm_count = torch.cuda.get_device_properties(a.device).multi_processor_count
    tiles = batch * triton.cdiv(m, 128) * triton.cdiv(n, 128)
    bmm_kernel_persistent[(min(sm_count, tiles),)](
        a,
        b,
        out,
        batch,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        NUM_SMS=sm_count,
        num_stages=3,
        num_warps=8,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=out.numel() > 2**31,
    )
