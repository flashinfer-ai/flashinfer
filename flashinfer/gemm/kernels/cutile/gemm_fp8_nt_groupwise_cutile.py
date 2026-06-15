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

* No TMA variant in v1 — the non-TMA path is simpler to verify. A TMA
  follow-up will be a separate MR once the baseline is reviewed.
"""

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search


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
):
    """Launch W8A8 FP8 matmul kernel with exhaustive_search autotuning."""
    cache_key = (
        M,
        N,
        K,
        block_n,
        block_k,
        output_dtype_int,
        A.dtype,
        str(A.device),
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

        def args_fn(cfg):
            return (
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
                cfg.BLOCK_SIZE_M,
                cfg.BLOCK_SIZE_N,
                cfg.BLOCK_SIZE_K,
                cfg.GROUP_SIZE_M,
                output_dtype_int,
                int(cfg.swap_ab),
            )

        def hints_fn(cfg):
            return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

        result = exhaustive_search(
            configs,
            stream,
            grid_fn,
            _w8a8_block_fp8_matmul_kernel,
            args_fn,
            hints_fn,
        )
        best_cfg = result.best.config
        tuned_kernel = ct.kernel(
            _w8a8_block_fp8_matmul_kernel._pyfunc,
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
        (
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
            best_cfg.BLOCK_SIZE_M,
            best_cfg.BLOCK_SIZE_N,
            best_cfg.BLOCK_SIZE_K,
            best_cfg.GROUP_SIZE_M,
            output_dtype_int,
            int(best_cfg.swap_ab),
        ),
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


def group_gemm_fp8_nt_groupwise_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: torch.Tensor,
    scale_granularity_mnk: tuple = (1, 128, 128),
    scale_major_mode: str = "K",
) -> torch.Tensor:
    """Group GEMM with FP8 block-scaled inputs via cuTile.

    Iterates over groups defined by ``m_indptr`` and dispatches each group
    to :func:`gemm_fp8_nt_groupwise_cutile`.

    Parameters
    ----------
    a : (cum_m, k) FP8 e4m3, row-major.
    b : (batch_size, n, k) FP8 e4m3, row-major.
    a_scale : (cum_m, k // block_k) float32, K-major scale for ``a``.
    b_scale : (batch_size, n // block_n, k // block_k) float32, K-major scale for ``b``.
    m_indptr : (batch_size + 1,) int32 row-segment boundaries.
    out : (cum_m, n) output buffer; written in place per group.
    scale_granularity_mnk : (m_g, n_g, k_g) — forwarded to each group call.
    scale_major_mode : must be ``"K"`` (the only mode supported by the cuTile kernel).

    Returns
    -------
    The same ``out`` tensor.
    """
    num_groups = m_indptr.shape[0] - 1
    m_indptr_cpu = m_indptr.cpu()

    for i in range(num_groups):
        m_start = int(m_indptr_cpu[i])
        m_end = int(m_indptr_cpu[i + 1])
        if m_start == m_end:
            continue

        a_i = a[m_start:m_end].contiguous()
        b_i = b[i].contiguous()
        a_scale_i = a_scale[m_start:m_end].contiguous()
        b_scale_i = b_scale[i].contiguous()
        out_i = out[m_start:m_end]

        gemm_fp8_nt_groupwise_cutile(
            a=a_i,
            b=b_i,
            a_scale=a_scale_i,
            b_scale=b_scale_i,
            out=out_i,
            scale_granularity_mnk=scale_granularity_mnk,
            scale_major_mode=scale_major_mode,
        )

    return out
