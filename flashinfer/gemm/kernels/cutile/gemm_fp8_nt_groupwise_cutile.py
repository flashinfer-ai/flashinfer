# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile (cuda.tile Python) FP8 block-scaled GEMM for FlashInfer.

This module provides ``gemm_fp8_nt_groupwise_cutile`` — a block-scaled W8A8
FP8 GEMM that plugs into the existing ``flashinfer.gemm.gemm_base.gemm_fp8_nt_groupwise``
dispatcher alongside the ``cutlass`` and ``trtllm`` backends. It targets the
DeepSeek-R1 / DeepSeek-V3 FP8 inference hot path:

    out = dequant(a @ b.T) where
        a is (M, K) FP8 e4m3 row-major
        b is (N, K) FP8 e4m3 row-major (column-major view as upstream caller
            passes)
        a_scale is (M, K // block_k) FP32 K-major (per-token-group scale)
        b_scale is (N // block_n, K // block_k) FP32 (per-block scale)
    with block_n = block_k = 128 by default
    (i.e. scale_granularity_mnk = (1, 128, 128), scale_major_mode = "K").

The cuTile kernel and autotune logic are ported verbatim from NVIDIA TileGym
(https://github.com/NVIDIA/TileGym), specifically
``src/tilegym/ops/cutile/fp8_quantization_matmul.py``. TileGym-internal
decorators (``@register_impl``) and helpers (``cached_replace_hints``,
``mark_perf_ready``) are stripped in favor of equivalent public
``cuda.tile`` APIs so this module has no TileGym runtime dependency.

Lessons applied from the BF16 cuTile port (MR adding ``mm_bf16(cutile)``):

* ``from __future__ import annotations`` is NOT used — it would convert the
  ``ct.Constant[int]`` annotations into strings at function-definition time
  and break ``cuda.tile``'s runtime introspection of the
  ``Annotated[int, ConstantAnnotation()]`` metadata.

* ``out.zero_()`` is called before the kernel launch. The W8A8 kernel uses
  ``ct.scatter`` to write outputs (not a load-and-blend), so it does not
  have the ``0 * NaN = NaN`` epilogue trap that the alpha-beta kernel has;
  the zeroing here is a defensive consistency measure to make the cuTile
  family behave uniformly with respect to uninitialized output buffers.

* Two kernel variants live here: ``_w8a8_block_fp8_matmul_kernel`` (the v1
  ``ct.gather``/``ct.scatter`` baseline, kept for verification/fallback) and
  ``_w8a8_block_fp8_matmul_kernel_tma`` (tiled ``ct.load``/``ct.store``, TMA).
  The launcher defaults to the TMA kernel (``use_tma=True``); it is the perf
  path that lifts the gather baseline (~0.36x cutlass) toward parity, matching
  TileGym's production ``w8a8_block_fp8_matmul_kernel_ct_tma``.
"""

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from ....cutile.cutile_common import cached_replace_hints


# Module-level tune cache:
#   key:   (M, N, K, block_n, block_k, output_dtype_int, dtype, str(device))
#   value: (best_cfg, kernel bound to chosen num_ctas/occupancy)
_W8A8_TUNE_CACHE: dict = {}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _gemm_calculate_pid_ct(pid, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M):
    """Swizzle linear block id into (pid_m, pid_n) for L2 cache locality."""
    num_pid_m = ct.cdiv(M, BLOCK_M)
    num_pid_n = ct.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# Ported verbatim from NVIDIA TileGym
# (https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/fp8_quantization_matmul.py).
@ct.kernel
def _w8a8_block_fp8_matmul_kernel(
    # Tensors
    A,
    B,
    C,
    As,
    Bs,
    # Dimensions
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int],
    # Quantization block sizes
    GROUP_N: ct.Constant[int],
    GROUP_K: ct.Constant[int],
    # Strides
    STRIDE_AM: ct.Constant[int],
    STRIDE_AK: ct.Constant[int],
    STRIDE_BK: ct.Constant[int],
    STRIDE_BN: ct.Constant[int],
    STRIDE_CM: ct.Constant[int],
    STRIDE_CN: ct.Constant[int],
    STRIDE__AS_M: ct.Constant[int],
    STRIDE__AS_K: ct.Constant[int],
    STRIDE__BS_K: ct.Constant[int],
    STRIDE__BS_N: ct.Constant[int],
    # Tile parameters
    BLOCK_SIZE_M: ct.Constant[int],
    BLOCK_SIZE_N: ct.Constant[int],
    BLOCK_SIZE_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    OUTPUT_DTYPE: ct.Constant[int],
    SWAP_AB: ct.Constant[int],
):
    """Gather/scatter W8A8 block-scaled FP8 matmul.

    When swap_ab=1: compute (B @ A^T)^T * scales  (swap operand order).
    When swap_ab=0: compute (A @ B^T) * scales     (normal order).

    A: (M, K)  B: (N, K)  As: (M, K_groups)  Bs: (N_groups, K_groups)  C: (M, N)

    Requires BLOCK_SIZE_N == group_n and BLOCK_SIZE_K == group_k for correct
    scale indexing (one scale per tile).
    """
    ct.static_assert(
        BLOCK_SIZE_N == GROUP_N,
        f"Kernel requires BLOCK_SIZE_N == group_n, got {BLOCK_SIZE_N} vs {GROUP_N}",
    )
    ct.static_assert(
        BLOCK_SIZE_K == GROUP_K,
        f"Kernel requires BLOCK_SIZE_K == group_k, got {BLOCK_SIZE_K} vs {GROUP_K}",
    )

    pid = ct.bid(0)
    pid_m, pid_n = _gemm_calculate_pid_ct(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    offs_am = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
    offs_bn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
    offs_k_base = ct.arange(BLOCK_SIZE_K, dtype=ct.int32)

    accumulator = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)

    num_k_tiles = ct.cdiv(K, BLOCK_SIZE_K)
    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_SIZE_K
        offs_k = offs_k_base + k_start

        # Load A block: (M, K) -> (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = ct.gather(
            A,
            (offs_am[:, None], offs_k[None, :]),
            check_bounds=True,
            padding_value=ct.float8_e4m3fn(0.0),
        )

        # Load B block: (N, K) -> (BLOCK_SIZE_N, BLOCK_SIZE_K)
        b = ct.gather(
            B,
            (offs_bn[:, None], offs_k[None, :]),
            check_bounds=True,
            padding_value=ct.float8_e4m3fn(0.0),
        )

        # As: (M, K_groups) -> (BLOCK_SIZE_M,)
        a_s = ct.gather(As, (offs_am, k_tile), check_bounds=True, padding_value=0.0)

        # Bs: (N_groups, K_groups) -> scalar
        b_s = ct.gather(Bs, (pid_n, k_tile), check_bounds=True, padding_value=0.0)
        ab_s = a_s[:, None] * b_s

        # MMA with permute for transpose
        if SWAP_AB:
            zero_acc = ct.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=ct.float32)
            a_t = ct.permute(a, (1, 0))
            dot_result = ct.mma(b, a_t, acc=zero_acc)
            dot_result = ct.permute(dot_result, (1, 0))
        else:
            zero_acc = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)
            b_t = ct.permute(b, (1, 0))
            dot_result = ct.mma(a, b_t, acc=zero_acc)

        accumulator = accumulator + dot_result * ab_s

    # Cast to output dtype
    if OUTPUT_DTYPE == 0:  # torch.float32
        c = accumulator
    elif OUTPUT_DTYPE == 1:  # torch.float16
        c = ct.astype(accumulator, ct.float16)
    elif OUTPUT_DTYPE == 2:  # torch.bfloat16
        c = ct.astype(accumulator, ct.bfloat16)
    else:
        c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
    offs_cn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
    ct.scatter(C, (offs_cm[:, None], offs_cn[None, :]), c, check_bounds=True)


# TMA-optimized variant, ported from NVIDIA TileGym's production kernel
# ``w8a8_block_fp8_matmul_kernel_ct_tma``
# (src/tilegym/suites/unsloth/cutile/fp8.py). The gather kernel above was the
# v1 baseline (simple to verify); this is the perf path.
@ct.kernel
def _w8a8_block_fp8_matmul_kernel_tma(
    # Tensors
    A,
    B,
    C,
    As,
    Bs,
    # Dimensions
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int],
    # Quantization block sizes
    GROUP_N: ct.Constant[int],
    GROUP_K: ct.Constant[int],
    # Tile parameters
    BLOCK_SIZE_M: ct.Constant[int],
    BLOCK_SIZE_N: ct.Constant[int],
    BLOCK_SIZE_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    OUTPUT_DTYPE: ct.Constant[int],
    SWAP_AB: ct.Constant[int],
):
    """TMA-capable W8A8 block-scaled FP8 matmul.

    Numerically identical to :func:`_w8a8_block_fp8_matmul_kernel` (same
    swizzle, MMA, scale application, epilogue), but loads the A/B tiles and
    stores the C tile via the tiled ``ct.load`` / ``ct.store`` path instead of
    ``ct.gather`` / ``ct.scatter``. That path lowers to TMA and is what lifts
    this kernel from the gather baseline (~0.36x cutlass) toward parity. The
    load idiom (``index`` / ``shape`` / ``order`` / ``padding_mode=ZERO`` /
    ``latency``) mirrors ``mm_bf16_cutile`` so out-of-tile M/N/K are zero-filled
    on load and clipped on store — arbitrary shapes stay correct.

    Per-block scales stay on ``ct.gather``: they are too small for TMA
    (contig_dim * elem_size < 16 B), matching the TileGym source.

    Requires BLOCK_SIZE_N == GROUP_N and BLOCK_SIZE_K == GROUP_K (one scale per
    N-tile / K-tile).
    """
    ct.static_assert(
        BLOCK_SIZE_N == GROUP_N,
        f"Kernel requires BLOCK_SIZE_N == group_n, got {BLOCK_SIZE_N} vs {GROUP_N}",
    )
    ct.static_assert(
        BLOCK_SIZE_K == GROUP_K,
        f"Kernel requires BLOCK_SIZE_K == group_k, got {BLOCK_SIZE_K} vs {GROUP_K}",
    )

    zero_pad = ct.PaddingMode.ZERO

    pid = ct.bid(0)
    pid_m, pid_n = _gemm_calculate_pid_ct(
        pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Row indices for the per-token-group activation-scale gather.
    offs_am = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)

    accumulator = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)

    num_k_tiles = ct.cdiv(K, BLOCK_SIZE_K)
    for k_tile in range(num_k_tiles):
        # A: (M, K) -> tile (pid_m, k_tile) of (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = ct.load(
            A,
            index=(pid_m, k_tile),
            shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(0, 1),
            padding_mode=zero_pad,
            latency=3,
        )
        # B: (N, K) -> tile (pid_n, k_tile) of (BLOCK_SIZE_N, BLOCK_SIZE_K)
        b = ct.load(
            B,
            index=(pid_n, k_tile),
            shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
            order=(0, 1),
            padding_mode=zero_pad,
            latency=3,
        )

        # Per-block scales via gather (too small for TMA); latency=4 prefetches
        # them ahead of the MMA, matching the TileGym source.
        # As: (M, K_groups) -> (BLOCK_SIZE_M,); Bs: (N_groups, K_groups) -> scalar
        a_s = ct.gather(
            As, (offs_am, k_tile), check_bounds=True, padding_value=0.0, latency=4
        )
        b_s = ct.gather(
            Bs, (pid_n, k_tile), check_bounds=True, padding_value=0.0, latency=4
        )
        ab_s = a_s[:, None] * b_s

        # MMA with permute for transpose
        if SWAP_AB:
            zero_acc = ct.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=ct.float32)
            a_t = ct.permute(a, (1, 0))
            dot_result = ct.mma(b, a_t, acc=zero_acc)
            dot_result = ct.permute(dot_result, (1, 0))
        else:
            zero_acc = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)
            b_t = ct.permute(b, (1, 0))
            dot_result = ct.mma(a, b_t, acc=zero_acc)

        accumulator = accumulator + dot_result * ab_s

    # Cast to output dtype
    if OUTPUT_DTYPE == 0:  # torch.float32
        c = accumulator
    elif OUTPUT_DTYPE == 1:  # torch.float16
        c = ct.astype(accumulator, ct.float16)
    elif OUTPUT_DTYPE == 2:  # torch.bfloat16
        c = ct.astype(accumulator, ct.bfloat16)
    else:
        c = accumulator

    ct.store(C, index=(pid_m, pid_n), tile=c, order=(0, 1))


def _w8a8_autotune_configs(block_n_quant, block_k_quant):
    """Yield autotune configurations for the W8A8 FP8 matmul kernel.

    ``BLOCK_SIZE_N`` and ``BLOCK_SIZE_K`` must equal the quantization block
    sizes for correct scale indexing (one scale per tile), so only
    ``BLOCK_SIZE_M``, ``occupancy``, and ``swap_ab`` are searched.
    """
    for block_m in [16, 32, 64, 128]:
        for occupancy in [1, 2, 4]:
            for swap_ab in [True, False]:
                yield SimpleNamespace(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n_quant,
                    BLOCK_SIZE_K=block_k_quant,
                    GROUP_SIZE_M=16,
                    num_ctas=1,
                    occupancy=occupancy,
                    swap_ab=swap_ab,
                )


def _w8a8_early_config_prune(configs, M):
    """Drop configs whose BLOCK_SIZE_M exceeds the M dimension."""
    pruned = [cfg for cfg in configs if cfg.BLOCK_SIZE_M <= M]
    return pruned if pruned else configs


def _w8a8_autotune_and_launch(
    stream,
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    block_n,
    block_k,
    output_dtype_int,
    use_tma=True,
):
    """Launch W8A8 FP8 matmul kernel with exhaustive_search autotuning.

    ``use_tma`` selects the tiled ``ct.load``/``ct.store`` (TMA) kernel — the
    default perf path — over the ``ct.gather``/``ct.scatter`` baseline. The two
    kernels differ only in the load/store mechanism (identical math), but the
    TMA kernel takes no stride arguments, so its launch arg tuple is shorter;
    ``use_tma`` is part of the cache key so tuned results never cross over.
    """
    kernel = (
        _w8a8_block_fp8_matmul_kernel_tma
        if use_tma
        else _w8a8_block_fp8_matmul_kernel
    )

    def build_args(cfg):
        head = (A, B, C, As, Bs, M, N, K, block_n, block_k)
        tail = (
            cfg.BLOCK_SIZE_M,
            cfg.BLOCK_SIZE_N,
            cfg.BLOCK_SIZE_K,
            cfg.GROUP_SIZE_M,
            output_dtype_int,
            int(cfg.swap_ab),
        )
        if use_tma:
            return head + tail
        strides = (
            A.stride(-2),
            A.stride(-1),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            As.stride(-2),
            As.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
        )
        return head + strides + tail

    cache_key = (
        M,
        N,
        K,
        block_n,
        block_k,
        output_dtype_int,
        A.dtype,
        str(A.device),
        use_tma,
    )

    if cache_key not in _W8A8_TUNE_CACHE:
        configs = _w8a8_early_config_prune(
            list(_w8a8_autotune_configs(block_n, block_k)),
            M,
        )

        def grid_fn(cfg):
            grid_m = _cdiv(M, cfg.BLOCK_SIZE_M)
            grid_n = _cdiv(N, cfg.BLOCK_SIZE_N)
            return (grid_m * grid_n, 1, 1)

        def hints_fn(cfg):
            return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

        result = exhaustive_search(
            configs,
            stream,
            grid_fn,
            kernel,
            build_args,
            hints_fn,
        )
        best_cfg = result.best.config
        tuned_kernel = ct.kernel(
            kernel._pyfunc,
            num_ctas=best_cfg.num_ctas,
            occupancy=best_cfg.occupancy,
        )
        _W8A8_TUNE_CACHE[cache_key] = (best_cfg, tuned_kernel)

    best_cfg, tuned_kernel = _W8A8_TUNE_CACHE[cache_key]
    grid_m = _cdiv(M, best_cfg.BLOCK_SIZE_M)
    grid_n = _cdiv(N, best_cfg.BLOCK_SIZE_N)
    ct.launch(
        stream,
        (grid_m * grid_n, 1, 1),
        tuned_kernel,
        build_args(best_cfg),
    )


_DTYPE_INT_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def gemm_fp8_nt_groupwise_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    scale_granularity_mnk: tuple = (1, 128, 128),
    scale_major_mode: str = "K",
) -> torch.Tensor:
    """BF16/FP16/FP32-out FP8 block-scaled GEMM via cuTile.

    Computes ``out = a @ b.T`` where ``a`` is FP8 e4m3 (M, K) row-major,
    ``b`` is FP8 e4m3 (N, K) row-major, and the scales ``a_scale`` /
    ``b_scale`` are applied per-block.

    Parameters
    ----------
    a : (M, K) FP8 e4m3, row-major, contiguous.
    b : (N, K) FP8 e4m3, row-major, contiguous.
    a_scale : per-token-group scale for ``a``, shape (M, K // block_k),
        K-major (``scale_major_mode == "K"``).
    b_scale : per-block scale for ``b``, shape (N // block_n, K // block_k),
        K-major.
    out : (M, N) output buffer, bf16/fp16/fp32; must be contiguous.
    scale_granularity_mnk : (m_g, n_g, k_g). The kernel currently supports
        ``m_g == 1`` (per-token-group on M); ``n_g`` becomes ``block_n`` and
        ``k_g`` becomes ``block_k``.
    scale_major_mode : currently only ``"K"`` is supported. The cuTile
        kernel reads ``As[m, k_group]`` and ``Bs[n_group, k_group]`` in that
        layout. ``"MN"`` mode would require an additional transpose adapter
        on the scales — left as a follow-up if the cutlass/trtllm paths
        actually exercise it.

    Returns
    -------
    The same ``out`` tensor (modified in place).
    """
    if scale_major_mode != "K":
        raise NotImplementedError(
            f"cuTile gemm_fp8_nt_groupwise only supports scale_major_mode='K' "
            f"in v1; got {scale_major_mode!r}. MN-major scale support is a "
            f"follow-up."
        )

    m_g, n_g, k_g = scale_granularity_mnk
    if m_g != 1:
        raise NotImplementedError(
            f"cuTile gemm_fp8_nt_groupwise requires scale_granularity_mnk[0] == 1 "
            f"(per-token-group on M); got {m_g}."
        )
    block_n, block_k = n_g, k_g

    # Defensive zero of the output: the cuTile family uniformly does
    # out.zero_() in its public entries to guard against propagating NaN/Inf
    # from uninitialized output storage (the alpha-beta family hits this
    # through the 0 * c_load term; for this kernel the ct.scatter epilogue
    # is a pure write so it's not strictly required, but zeroing keeps the
    # behaviour consistent across cuTile entries and removes a class of
    # surprising bugs).
    out.zero_()

    # Shape sanity checks — use explicit ValueErrors instead of `assert` so
    # the validation isn't elided when Python is run with `-O` (which strips
    # assert statements and would let bad inputs reach the cuda.tile kernel).
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("a and b must be contiguous")
    if not (a.dim() == 2 and b.dim() == 2):
        raise ValueError("a and b must be 2D")
    M, KA = a.shape
    N, KB = b.shape
    if KA != KB:
        raise ValueError(f"a.shape[-1] ({KA}) must match b.shape[-1] ({KB})")
    K = KA
    if out.shape != (M, N):
        raise ValueError(f"out must be ({M},{N}); got {tuple(out.shape)}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if not (a_scale.dim() == 2 and b_scale.dim() == 2):
        raise ValueError("scales must be 2D")
    # a_scale shape: (M, K // block_k); b_scale shape: (N // block_n, K // block_k)
    if a_scale.shape != (M, _cdiv(K, block_k)):
        raise ValueError(
            f"a_scale must be ({M}, {_cdiv(K, block_k)}); got {tuple(a_scale.shape)}"
        )
    if b_scale.shape != (_cdiv(N, block_n), _cdiv(K, block_k)):
        raise ValueError(
            f"b_scale must be ({_cdiv(N, block_n)}, {_cdiv(K, block_k)}); "
            f"got {tuple(b_scale.shape)}"
        )

    out_dtype_int = _DTYPE_INT_MAP.get(out.dtype)
    if out_dtype_int is None:
        raise ValueError(
            f"out.dtype {out.dtype} not supported by cuTile gemm_fp8_nt_groupwise; "
            f"expected bf16 / fp16 / fp32"
        )

    # Pin the stream to ``a.device`` for multi-GPU correctness — same fix as
    # gemm.py / bmm.py.
    _w8a8_autotune_and_launch(
        torch.cuda.current_stream(a.device),
        a,
        b,
        out,
        a_scale,
        b_scale,
        M,
        N,
        K,
        block_n,
        block_k,
        out_dtype_int,
    )
    return out


@ct.kernel
def _w8a8_group_fp8_matmul_kernel(
    # Tensors
    A,  # (total_m, K) FP8
    B,  # (Q * N, K) FP8 — the (Q, N, K) weights flattened on the first two dims
    C,  # (total_m, N) output
    As,  # (total_m, K_groups) FP32
    Bs,  # (Q * N_groups, K_groups) FP32 — b_scale flattened on (Q, N_groups)
    m_indptr,  # (Q + 1,) int32 row-segment boundaries (prefix sums)
    # Dimensions
    Q: ct.Constant[int],
    TOTAL_M: ct.Constant[int],  # A/C row count == C.shape[0]; also the OOB store sentinel
    max_m_device,  # (1,) int32 device tensor: max over groups of (m_indptr[i+1]-m_indptr[i])
    N: ct.Constant[int],
    K: ct.Constant[int],
    GROUP_N: ct.Constant[int],
    GROUP_K: ct.Constant[int],
    # Tile parameters
    BLOCK_SIZE_M: ct.Constant[int],
    BLOCK_SIZE_N: ct.Constant[int],
    BLOCK_SIZE_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    OUTPUT_DTYPE: ct.Constant[int],
    SWAP_AB: ct.Constant[int],
):
    """Fused, CUDA-graph-safe grouped W8A8 block-scaled FP8 matmul.

    Computes, for each group ``q`` defined by ``m_indptr`` boundaries,
    ``C[rows_q] = (A[rows_q] @ B[q].T) * scales`` in a SINGLE persistent launch
    — replacing the old host loop that did a ``m_indptr.cpu()`` D2H sync + one
    launch per group (illegal under CUDA-graph capture, and launch-latency-bound
    for small groups).

    Design — the two constraints that shape this kernel:
    * **No host sync.** All segment boundaries are read on-device
      (``ct.load(m_indptr, ...)``); the persistent-loop tile count uses the
      device-side ``max_m_device`` (mirrors ragged_block_scaled_bmm's
      defense-in-depth). The grid is a fixed NUM_SMS multiple, independent of any
      per-group size, so the launch needs nothing host-side about ``m_indptr``.
    * **Arbitrary (non-aligned) segments.** Group sizes are arbitrary token
      counts (e.g. MoE), so A/C rows are addressed with ``ct.gather``/``ct.scatter``
      on runtime row offsets (``m_start + pid_m*BLOCK_M + arange``), NOT the tiled
      ct.load path that ragged_block_scaled_bmm uses (that needs BLOCK_M-aligned
      segments). A tile's tail rows can spill past ``m_end`` into the next group;
      the store index for those rows is set to the OOB sentinel ``TOTAL_M`` so
      ``ct.scatter(check_bounds=True)`` drops them (the row_valid->OOB idiom from
      TileGym's MoE grouped_gemm). The spilled rows' MMA output is simply unused.

    B / b_scale are gathered from the (Q, N, K) / (Q, N_groups, K_groups) tensors
    flattened on their leading two dims (row ``pid_q*N + offs_n`` / ``pid_q*N_groups
    + pid_n``): N and K are quant-block aligned, so no masking is needed there.

    Requires BLOCK_SIZE_N == GROUP_N and BLOCK_SIZE_K == GROUP_K (one scale per
    N-tile / K-tile), same as the single-GEMM kernels.
    """
    ct.static_assert(
        BLOCK_SIZE_N == GROUP_N,
        f"Kernel requires BLOCK_SIZE_N == group_n, got {BLOCK_SIZE_N} vs {GROUP_N}",
    )
    ct.static_assert(
        BLOCK_SIZE_K == GROUP_K,
        f"Kernel requires BLOCK_SIZE_K == group_k, got {BLOCK_SIZE_K} vs {GROUP_K}",
    )

    pid = ct.bid(0)

    # Persistent-loop bound from device-side ground truth (no host sync).
    max_m_runtime = ct.load(max_m_device, index=(0,), shape=(1,)).item()
    num_pid_m = ct.cdiv(max_m_runtime, BLOCK_SIZE_M)
    num_pid_n = ct.cdiv(N, BLOCK_SIZE_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * Q
    num_programs = ct.num_blocks(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_k_base = ct.arange(BLOCK_SIZE_K, dtype=ct.int32)

    for current_pid in range(pid, total_tiles, num_programs):
        pid_q = current_pid // tiles_per_batch
        pid_in_batch = current_pid % tiles_per_batch

        group_id = pid_in_batch // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid_in_batch % group_size_m_actual)
        pid_n = (pid_in_batch % num_pid_in_group) // group_size_m_actual

        m_start = ct.load(m_indptr, index=(pid_q,), shape=(1,)).item()
        m_end = ct.load(m_indptr, index=(pid_q + 1,), shape=(1,)).item()
        valid_m = m_end - m_start

        if pid_m * BLOCK_SIZE_M < valid_m:
            local_m = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
            offs_am = m_start + local_m  # global rows into flattened A / As / C
            row_valid = local_m < valid_m
            offs_bn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
            b_rows = pid_q * N + offs_bn  # rows into the (Q*N, K) flattened B
            bs_row = pid_q * num_pid_n + pid_n  # row into (Q*N_groups, K_groups) Bs

            accumulator = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)

            num_k_tiles = ct.cdiv(K, BLOCK_SIZE_K)
            for k_tile in range(num_k_tiles):
                offs_k = offs_k_base + k_tile * BLOCK_SIZE_K

                # A: gather this group's rows (arbitrary offset)
                a = ct.gather(
                    A,
                    (offs_am[:, None], offs_k[None, :]),
                    check_bounds=True,
                    padding_value=ct.float8_e4m3fn(0.0),
                )
                # B: gather group pid_q's (BLOCK_N, BLOCK_K) tile from flattened B
                b = ct.gather(
                    B,
                    (b_rows[:, None], offs_k[None, :]),
                    check_bounds=True,
                    padding_value=ct.float8_e4m3fn(0.0),
                )

                # As: (total_m, K_groups) -> (BLOCK_M,); Bs: scalar for (group, n-tile, k-tile)
                a_s = ct.gather(
                    As, (offs_am, k_tile), check_bounds=True, padding_value=0.0, latency=4
                )
                b_s = ct.gather(
                    Bs, (bs_row, k_tile), check_bounds=True, padding_value=0.0, latency=4
                )
                ab_s = a_s[:, None] * b_s

                # MMA with permute for transpose
                if SWAP_AB:
                    zero_acc = ct.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=ct.float32)
                    a_t = ct.permute(a, (1, 0))
                    dot_result = ct.mma(b, a_t, acc=zero_acc)
                    dot_result = ct.permute(dot_result, (1, 0))
                else:
                    zero_acc = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)
                    b_t = ct.permute(b, (1, 0))
                    dot_result = ct.mma(a, b_t, acc=zero_acc)

                accumulator = accumulator + dot_result * ab_s

            if OUTPUT_DTYPE == 1:  # torch.float16
                c = ct.astype(accumulator, ct.float16)
            elif OUTPUT_DTYPE == 2:  # torch.bfloat16
                c = ct.astype(accumulator, ct.bfloat16)
            else:  # torch.float32
                c = accumulator

            # Drop tail rows that spilled past this group (route to OOB row TOTAL_M).
            offs_cm = ct.where(row_valid, offs_am, TOTAL_M)
            ct.scatter(
                C, (offs_cm[:, None], offs_bn[None, :]), c, check_bounds=True
            )


def _group_gemm_default_config(total_m, num_groups):
    """Static kernel config for the fused grouped kernel (no exhaustive_search
    at call time — that would break CUDA-graph capture).

    BLOCK_SIZE_N / BLOCK_SIZE_K are locked to the quant block (128) for correct
    scale indexing, so only BLOCK_M / swap / occupancy vary. Small average-M
    groups take a swap_ab config with a smaller BLOCK_M (fewer wasted tile rows
    and better MMA shape); large groups take the straight BLOCK_M=128 path.
    """
    avg_m = total_m / max(num_groups, 1)
    if avg_m >= 256:
        return SimpleNamespace(
            BLOCK_SIZE_M=128, GROUP_SIZE_M=8, swap_ab=False, num_ctas=1, occupancy=1
        )
    return SimpleNamespace(
        BLOCK_SIZE_M=64, GROUP_SIZE_M=8, swap_ab=True, num_ctas=1, occupancy=1
    )


def group_gemm_fp8_nt_groupwise_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: torch.Tensor,
    scale_granularity_mnk: tuple = (1, 128, 128),
    scale_major_mode: str = "K",
    segment_alignment: int = 1,
) -> torch.Tensor:
    """Group GEMM with FP8 block-scaled inputs via cuTile — single fused launch.

    For every group ``q`` defined by ``m_indptr``, computes
    ``out[rows_q] = (a[rows_q] @ b[q].T)`` with per-block FP8 dequant scales,
    in one persistent :func:`_w8a8_group_fp8_matmul_kernel` launch. All segment
    boundaries are read on-device, so this is CUDA-graph-capturable — unlike the
    previous host loop, which did a ``m_indptr.cpu()`` D2H sync (illegal during
    capture) and one launch + one ``.contiguous()`` copy per group.

    Parameters
    ----------
    a : (cum_m, k) FP8 e4m3, row-major.
    b : (batch_size, n, k) FP8 e4m3, row-major.
    a_scale : (cum_m, k // block_k) float32, K-major scale for ``a``.
    b_scale : (batch_size, n // block_n, k // block_k) float32, K-major scale for ``b``.
    m_indptr : (batch_size + 1,) int32 row-segment boundaries (prefix sums).
    out : (cum_m, n) output buffer; written in place. Every row belongs to exactly
        one group and is written, so no pre-zeroing is required.
    scale_granularity_mnk : (m_g, n_g, k_g). Requires ``m_g == 1``; ``n_g`` becomes
        ``block_n`` and ``k_g`` becomes ``block_k``.
    scale_major_mode : must be ``"K"``.
    segment_alignment : row alignment the caller GUARANTEES for every ``m_indptr``
        segment offset. Default 1 (arbitrary) uses the gather kernel. Pass a
        multiple of 128 (segments padded to that many rows) to take the faster
        aligned-segment TMA path. Cannot be validated at runtime without a host
        sync that would break CUDA-graph capture, so it is a caller contract.

    Returns
    -------
    The same ``out`` tensor.
    """
    if scale_major_mode != "K":
        raise NotImplementedError(
            f"cuTile group_gemm_fp8_nt_groupwise only supports scale_major_mode='K'; "
            f"got {scale_major_mode!r}."
        )
    m_g, n_g, k_g = scale_granularity_mnk
    if m_g != 1:
        raise NotImplementedError(
            f"cuTile group_gemm_fp8_nt_groupwise requires scale_granularity_mnk[0] == 1; "
            f"got {m_g}."
        )
    block_n, block_k = n_g, k_g

    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("a and b must be contiguous")
    if a.dim() != 2:
        raise ValueError("a must be 2D (cum_m, k)")
    if b.dim() != 3:
        raise ValueError("b must be 3D (batch_size, n, k)")
    total_m, K = a.shape
    Q, N, KB = b.shape
    if K != KB:
        raise ValueError(f"a.shape[-1] ({K}) must match b.shape[-1] ({KB})")
    if m_indptr.dim() != 1 or m_indptr.shape[0] != Q + 1:
        raise ValueError(f"m_indptr must be 1D with {Q + 1} elements")
    if out.shape != (total_m, N):
        raise ValueError(f"out must be ({total_m},{N}); got {tuple(out.shape)}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if a_scale.dim() != 2 or a_scale.shape != (total_m, _cdiv(K, block_k)):
        raise ValueError(
            f"a_scale must be ({total_m}, {_cdiv(K, block_k)}); got {tuple(a_scale.shape)}"
        )
    n_groups = _cdiv(N, block_n)
    if b_scale.dim() != 3 or b_scale.shape != (Q, n_groups, _cdiv(K, block_k)):
        raise ValueError(
            f"b_scale must be ({Q}, {n_groups}, {_cdiv(K, block_k)}); "
            f"got {tuple(b_scale.shape)}"
        )
    out_dtype_int = _DTYPE_INT_MAP.get(out.dtype)
    if out_dtype_int is None:
        raise ValueError(
            f"out.dtype {out.dtype} not supported; expected bf16 / fp16 / fp32"
        )

    if not (a_scale.is_contiguous() and b_scale.is_contiguous()):
        raise ValueError("a_scale and b_scale must be contiguous")

    # m_indptr must be int32 for the on-device index loads. Cast on device (no sync).
    if m_indptr.dtype != torch.int32:
        m_indptr = m_indptr.to(torch.int32)

    # Device-side max group size — computed without a host sync, so the launch
    # stays CUDA-graph-capturable. The kernel reads this for its persistent bound.
    seg_sizes = m_indptr[1:] - m_indptr[:-1]
    max_m_device = seg_sizes.max().to(torch.int32).reshape(1)

    # TMA fast path (opt-in). The default fused kernel gathers A/C on runtime row
    # offsets so it handles ARBITRARY segment sizes, but gather is bandwidth-bound
    # (~0.3x cutlass). When the caller GUARANTEES every m_indptr segment offset is
    # a multiple of ``segment_alignment`` (>=128, the common MoE case where token
    # counts are padded), the aligned-segment TMA kernel in
    # ``ragged_block_scaled_bmm`` can address rows as a tiled ``m_start // BLOCK_M``
    # ct.load/ct.store (TMA) — much faster and already validated bit-identical to
    # native ``group_gemm_fp8_nt_groupwise(trtllm)``. It is graph-safe (reads the
    # device-side max_m_device for its loop bound; ``max_m=total_m`` is a safe host
    # grid overestimate capped at NUM_SMS). This cannot be auto-detected without a
    # host sync that would break capture, so it is a caller contract.
    if segment_alignment >= 128 and segment_alignment % 128 == 0:
        from .ragged_block_scaled_bmm_cutile import ragged_block_scaled_bmm

        # Pass ``out`` so the kernel stores into it directly (no extra copy).
        ragged_block_scaled_bmm(
            a,
            b,
            a_scale,
            b_scale,
            m_indptr,
            max_m=total_m,
            max_m_device=max_m_device,
            transpose_a=False,
            transpose_b=True,
            out_dtype=out.dtype,
            out=out,
            segment_alignment=segment_alignment,
        )
        return out

    # Flatten B / b_scale on their leading dims so the kernel can gather group
    # ``q``'s rows as ``q*N + offs_n`` / ``q*N_groups + pid_n`` (both contiguous).
    b_flat = b.reshape(Q * N, K)
    b_scale_flat = b_scale.reshape(Q * n_groups, _cdiv(K, block_k))

    cfg = _group_gemm_default_config(total_m, Q)

    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    # Persistent grid, independent of any per-group size (no host max_m needed).
    num_programs = max(1, (num_sms // cfg.num_ctas)) * cfg.occupancy

    kernel = cached_replace_hints(
        _w8a8_group_fp8_matmul_kernel,
        num_ctas=cfg.num_ctas,
        occupancy=cfg.occupancy,
    )

    ct.launch(
        torch.cuda.current_stream(a.device),
        (num_programs, 1, 1),
        kernel,
        (
            a,
            b_flat,
            out,
            a_scale,
            b_scale_flat,
            m_indptr,
            Q,
            total_m,
            max_m_device,
            N,
            K,
            block_n,
            block_k,
            cfg.BLOCK_SIZE_M,
            block_n,
            block_k,
            cfg.GROUP_SIZE_M,
            out_dtype_int,
            int(cfg.swap_ab),
        ),
    )

    return out
