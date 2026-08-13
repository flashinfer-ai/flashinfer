# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
FP8 Batched Matrix Multiplication (BMM) Wrapper for CuTe-DSL Kernels
====================================================================

Location: flashinfer/gemm/kernels/bmm_fp8_wrapper.py

This module provides the high-level wrapper for FP8 batched matrix multiplication
using CuTe-DSL kernels, supporting both Blackwell (SM100) and Rubin (SM107) architectures.

It handles:
- Autotuning configuration spaces (SM100_AUTOTUNE_CONFIGS, SM107_AUTOTUNE_CONFIGS)
- Kernel compilation and caching with symbolic M, N, K dimensions
- Configuration validation for different problem sizes
- Entry points for both default and autotuned execution
"""

import functools
from typing import Callable, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch

from flashinfer.utils import get_compute_capability
from flashinfer.cute_dsl.utils import torch_dtype_to_cutlass

from .bmm_fp8_blackwell import (
    PersistentDenseGemmKernel,
    bmm,
)


# =============================================================================
# Autotune Configuration Space
# =============================================================================
#
# Each configuration is a named tuple defining kernel parameters for autotuning.
# The autotuner will try each valid configuration and select the fastest one.
#
# SM100 (Blackwell) constraints (from check_mma_tiler_and_cluster_shape):
#   - mma_tiler_mn[0]: 64/128 (1-CTA) or 128/256 (2-CTA)
#   - mma_tiler_mn[1]: 32, 64, 96, ..., 256 (multiples of 32)
#   - cluster_shape: powers of 2, product <= 16
#   - cluster_shape[0] must be multiple of 2 if use_2cta_instrs=True
#
# SM107 (Rubin) constraints:
#   - FP8 only, K=64 requires M=128 (1-CTA) or M=256 (2-CTA)
#   - Bkeep-Breuse pattern: mma_tiler[0] = 2 * mma_inst_shape[0]

# SM100 (Blackwell) Configuration Space
# Format: (mma_tiler_mn, cluster_shape_mn, use_2cta_instrs, use_tma_store, swizzle_size, raster_along)
# IMPORTANT: Index 0 is the fallback config - must be the most reliable/tested configuration
SM100_AUTOTUNE_CONFIGS: Tuple[Tuple, ...] = (
    # Format: (mma_tiler_mn, cluster_shape_mn, use_2cta_instrs, use_tma_store, swizzle_size, raster_along)
    #
    # Default/fallback config - this is the same as bmm_fp8_cute_dsl's hardcoded default
    # Must be first (index 0) since tactic=-1 falls back to index 0
    ((128, 128), (2, 1), True, True, 1, "m"),  # Index 0: DEFAULT FALLBACK
    #
    # === 2-CTA configurations with TMA store (best for large problems) ===
    # Large tiles for high throughput
    ((256, 256), (2, 1), True, True, 1, "m"),
    ((256, 128), (2, 1), True, True, 1, "m"),
    ((128, 256), (2, 1), True, True, 1, "m"),
    # Larger cluster shapes for more parallelism
    ((256, 128), (2, 2), True, True, 1, "m"),
    ((128, 128), (2, 2), True, True, 1, "m"),
    ((128, 128), (4, 1), True, True, 1, "m"),
    # Raster along N variants (better for certain aspect ratios)
    ((256, 128), (2, 1), True, True, 1, "n"),
    ((128, 256), (2, 1), True, True, 1, "n"),
    #
    # === 1-CTA configurations with TMA store (better for smaller problems) ===
    ((128, 128), (1, 1), False, True, 1, "m"),
    ((128, 64), (1, 1), False, True, 1, "m"),
    ((64, 128), (1, 1), False, True, 1, "m"),
    ((64, 64), (1, 1), False, True, 1, "m"),
)

# SM107 (Rubin) Configuration Space
# Format: (mma_tiler, mma_inst_shape, cluster_shape_mn, use_2cta_instrs, use_tma_store, swizzle_size, raster_along)
# Note: For FP8, mma_inst_shape K-mode must be 32 or 64 (not 128)
SM107_AUTOTUNE_CONFIGS: Tuple[Tuple, ...] = (
    #
    # === 2-CTA with TMA store: B-keep/B-reuse patterns (best for large problems) ===
    # B-reuse: mma_tiler[2] = 2 * mma_inst_shape[2] for better B matrix reuse
    ((256, 256, 128), (256, 256, 64), (2, 1), True, True, 1, "m"),  # K=64, large tiles
    ((256, 128, 128), (256, 128, 64), (2, 1), True, True, 1, "m"),  # K=64, medium tiles
    ((256, 256, 64), (256, 256, 32), (2, 1), True, True, 1, "m"),  # K=32, large tiles
    # Non-Breuse patterns (mma_tiler = mma_inst_shape)
    ((256, 256, 64), (256, 256, 64), (2, 1), True, True, 1, "m"),  # K=64, non-Breuse
    # Larger cluster shapes
    ((256, 256, 128), (256, 256, 64), (2, 2), True, True, 1, "m"),  # K=64, 2x2 cluster
    ((256, 128, 128), (256, 128, 64), (4, 1), True, True, 1, "m"),  # K=64, 4x1 cluster
    # Raster along N variants
    ((256, 256, 128), (256, 256, 64), (2, 1), True, True, 1, "n"),  # K=64, raster N
    ((256, 128, 128), (256, 128, 64), (2, 1), True, True, 1, "n"),  # K=64, raster N
)


# =============================================================================
# TVM-FFI Compiled Kernel Cache Functions
# =============================================================================


def _sm107_gemm_kernel_cls():
    """Import the SM107 kernel lazily.

    It requires CuTe DSL >= 4.8 (``cutlass.utils.rubin_helpers``); importing at
    module scope would break FlashInfer on older DSL releases.
    """
    from .bmm_fp8_rubin import SM107PersistentDenseGemmKernel

    return SM107PersistentDenseGemmKernel


def _get_stride_order(major: str, tensor_type: str) -> Tuple[int, int, int]:
    """Get stride order for make_fake_compact_tensor based on major dimension.

    For 3D tensors (batch, dim0, dim1), stride_order specifies which dimensions
    are contiguous. Lower values = more contiguous (stride 1).

    :param major: The major (contiguous) dimension name
    :param tensor_type: "a" for (batch, m, k), "b" for (batch, k, n), "c" for (batch, m, n)
    :return: stride_order tuple for make_fake_compact_tensor
    """
    # stride_order: lower index = more contiguous
    # For row-major (last dim contiguous): (2, 1, 0) means dim2 has stride 1
    # For col-major (middle dim contiguous): (2, 0, 1) means dim1 has stride 1
    if tensor_type == "a":
        # A: (batch, m, k) - k-major means k is contiguous
        return (2, 1, 0) if major == "k" else (2, 0, 1)
    elif tensor_type == "b":
        # B: (batch, k, n) - n-major means n is contiguous
        return (2, 1, 0) if major == "n" else (2, 0, 1)
    else:  # tensor_type == "c"
        # C: (batch, m, n) - n-major means n is contiguous
        return (2, 1, 0) if major == "n" else (2, 0, 1)


def _create_fake_tensors(
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
) -> Tuple:
    """Create fake tensors with symbolic M, N, K dimensions for kernel compilation.

    The M, N, and K dimensions are symbolic, allowing a single compiled kernel to
    handle any M, N, K values at runtime. This is important for LLM inference where
    M (sequence length) varies per request and avoids recompilation when N or K change.

    :return: Tuple of (a_fake, b_fake, c_fake, scale_fake, stream_fake)
    """
    a_stride_order = _get_stride_order(a_major, "a")
    b_stride_order = _get_stride_order(b_major, "b")
    c_stride_order = _get_stride_order(c_major, "c")

    # Use symbolic M, N, K dimensions - allows single kernel to handle any size at runtime
    sym_m = cute.sym_int()
    sym_n = cute.sym_int()
    sym_k = cute.sym_int()
    sym_bs = cute.sym_int()

    a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (sym_bs, sym_m, sym_k),
        stride_order=a_stride_order,
        assumed_align=16,
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (sym_bs, sym_k, sym_n),
        stride_order=b_stride_order,
        assumed_align=16,
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        c_dtype,
        (sym_bs, sym_m, sym_n),
        stride_order=c_stride_order,
        assumed_align=16,
    )

    # Create fake scale tensor for CUDA graph compatibility
    scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )

    # Create fake stream that uses environment stream at runtime
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return a_fake, b_fake, c_fake, scale_fake, stream_fake


def _compile_and_create_tensor_api(
    gemm_kernel,
    a_fake,
    b_fake,
    c_fake,
    scale_fake,
    stream_fake,
    cluster_shape_mn: Tuple[int, int],
) -> Callable:
    """Compile a GEMM kernel and return a tensor API closure.

    :param gemm_kernel: The GEMM kernel instance (PersistentDenseGemmKernel or SM107PersistentDenseGemmKernel)
    :param a_fake, b_fake, c_fake, scale_fake, stream_fake: Fake tensors for compilation
    :param cluster_shape_mn: Cluster shape for computing max active clusters
    :return: A closure that accepts torch tensors directly via TVM-FFI
    """
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    compiled_kernel = cute.compile(
        bmm,
        gemm_kernel,
        a_fake,
        b_fake,
        c_fake,
        max_active_clusters,
        stream_fake,
        scale_fake,
        lambda x: x,  # identity epilogue
        options="--enable-tvm-ffi",
    )

    # Create tensor API closure
    def tensor_api(
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        output_scale_tensor: torch.Tensor,
    ) -> None:
        """Runtime API that passes torch tensors directly via TVM-FFI.

        The output_scale_tensor is a 1-element Float32 tensor on GPU, avoiding
        host-device synchronization that would break CUDA graph capture.
        """
        # TVM-FFI handles FP8 dtype conversion internally
        # Scale is passed as a tensor and dereferenced on GPU
        compiled_kernel(
            a_tensor,
            b_tensor,
            c_tensor,
            output_scale_tensor,
        )

    return tensor_api


@functools.cache
def _get_compiled_bmm_sm100(
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    use_tma_store: bool,
    swizzle_size: int,
    raster_along: Literal["m", "n"],
) -> Callable:
    """Get compiled BMM kernel for SM100 using TVM-FFI.

    Returns a closure that accepts torch tensors directly without
    per-call tensor wrapping overhead.
    """
    a_fake, b_fake, c_fake, scale_fake, stream_fake = _create_fake_tensors(
        ab_dtype, c_dtype, a_major, b_major, c_major
    )

    gemm = PersistentDenseGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
        swizzle_size,
        raster_along,
    )

    return _compile_and_create_tensor_api(
        gemm, a_fake, b_fake, c_fake, scale_fake, stream_fake, cluster_shape_mn
    )


@functools.cache
def _get_compiled_bmm_sm107(
    batch: int,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler: Tuple[int, int, int],
    mma_inst_shape: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    use_tma_store: bool,
    swizzle_size: int,
    raster_along: Literal["m", "n"],
) -> Callable:
    """Get compiled BMM kernel for SM107 using TVM-FFI.

    Returns a closure that accepts torch tensors directly without
    per-call tensor wrapping overhead.
    """
    a_fake, b_fake, c_fake, scale_fake, stream_fake = _create_fake_tensors(
        ab_dtype, c_dtype, a_major, b_major, c_major
    )

    gemm = _sm107_gemm_kernel_cls()(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler,
        mma_inst_shape,
        cluster_shape_mn,
        use_tma_store,
        swizzle_size,
        raster_along,
    )

    return _compile_and_create_tensor_api(
        gemm, a_fake, b_fake, c_fake, scale_fake, stream_fake, cluster_shape_mn
    )


# =============================================================================
# Compile and Run Functions (Simplified)
# =============================================================================


def _compile_and_run_bmm_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    c_tensor: torch.Tensor,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    use_tma_store: bool,
    swizzle_size: int,
    raster_along: Literal["m", "n"],
    output_scale_tensor: torch.Tensor,
):
    """Compile (if needed) and run the BMM kernel for SM100.

    :param output_scale_tensor: 1-element Float32 tensor on GPU containing the scale.
                                For FP8 GEMM, this is typically a_scale * b_scale.
                                Using a tensor avoids host-device sync for CUDA graph compatibility.
    """
    batch, m, k = a_tensor.shape

    # Get cached compiled kernel (M, N, K are symbolic - kernel handles any size at runtime)
    # Only batch, dtypes, layouts, and config params are in the cache key
    # Note: cutlass types are hashable but mypy doesn't recognize this
    tensor_api = _get_compiled_bmm_sm100(
        batch,
        ab_dtype,  # type: ignore[arg-type]
        c_dtype,  # type: ignore[arg-type]
        acc_dtype,  # type: ignore[arg-type]
        a_major,
        b_major,
        c_major,
        mma_tiler_mn,
        cluster_shape_mn,
        use_2cta_instrs,
        use_tma_store,
        swizzle_size,
        raster_along,
    )
    # Run kernel - tensors passed directly via TVM-FFI
    # Actual M, N, K dimensions are determined at runtime from tensor shapes
    tensor_api(a_tensor, b_tensor, c_tensor, output_scale_tensor)


def _compile_and_run_bmm_sm107(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    c_tensor: torch.Tensor,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler: Tuple[int, int, int],
    mma_inst_shape: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    use_tma_store: bool,
    swizzle_size: int,
    raster_along: Literal["m", "n"],
    output_scale_tensor: torch.Tensor,
):
    """Compile (if needed) and run the BMM kernel for SM107.

    :param output_scale_tensor: 1-element Float32 tensor on GPU containing the scale.
                                For FP8 GEMM, this is typically a_scale * b_scale.
                                Using a tensor avoids host-device sync for CUDA graph compatibility.
    """
    batch, m, k = a_tensor.shape

    # Get cached compiled kernel (M, N, K are symbolic - kernel handles any size at runtime)
    # Only batch, dtypes, layouts, and config params are in the cache key
    # Note: cutlass types are hashable but mypy doesn't recognize this
    tensor_api = _get_compiled_bmm_sm107(
        batch,
        ab_dtype,  # type: ignore[arg-type]
        c_dtype,  # type: ignore[arg-type]
        acc_dtype,  # type: ignore[arg-type]
        a_major,
        b_major,
        c_major,
        mma_tiler,
        mma_inst_shape,
        cluster_shape_mn,
        use_2cta_instrs,
        use_tma_store,
        swizzle_size,
        raster_along,
    )

    # Run kernel - tensors passed directly via TVM-FFI
    # Actual M, N, K dimensions are determined at runtime from tensor shapes
    tensor_api(a_tensor, b_tensor, c_tensor, output_scale_tensor)


def _detect_major_dim(tensor: torch.Tensor, dim_names: Tuple[str, str]) -> str:
    """Detect the major dimension based on tensor strides.

    For a 3D tensor (batch, dim0, dim1), determine if dim0 or dim1 is the major
    (contiguous) dimension based on strides.

    :param tensor: 3D tensor
    :param dim_names: Tuple of (name_for_dim1, name_for_dim2), e.g., ("m", "k") for A
    :return: The name of the major dimension
    """
    strides = tensor.stride()
    # dim 1 is major if its stride is 1
    if strides[1] == 1:
        return dim_names[0]
    # dim 2 is major if its stride is 1
    elif strides[2] == 1:
        return dim_names[1]
    else:
        # Fallback: assume the dimension with smaller stride is major
        if strides[1] < strides[2]:
            return dim_names[0]
        else:
            return dim_names[1]


# =============================================================================
# Configuration Validation for Autotuning
# =============================================================================


def _can_implement_config_sm100(
    m: int,
    n: int,
    k: int,
    batch: int,
    config: Tuple,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
) -> bool:
    """Check if an SM100 config can implement the given problem size.

    :param m: M dimension
    :param n: N dimension
    :param k: K dimension
    :param batch: Batch size
    :param config: SM100 config tuple (mma_tiler_mn, cluster_shape_mn, use_2cta_instrs,
                   use_tma_store, swizzle_size, raster_along)
    :param ab_dtype: Input data type (cutlass.Numeric)
    :param c_dtype: Output data type (cutlass.Numeric)
    :param a_major: Major dimension of A ("m" or "k")
    :param b_major: Major dimension of B ("k" or "n")
    :param c_major: Major dimension of C ("m" or "n")
    :return: True if config can implement, False otherwise
    """
    (
        mma_tiler_mn,
        cluster_shape_mn,
        use_2cta_instrs,
        use_tma_store,
        swizzle_size,
        raster_along,
    ) = config

    # Additional minimum size check: problem must be at least as large as CTA tile
    # to avoid launching kernels with invalid grid dimensions
    cta_tile_m = mma_tiler_mn[0] // (2 if use_2cta_instrs else 1)
    cta_tile_n = mma_tiler_mn[1]

    # Problem size must be able to fill at least one CTA tile
    # This prevents "illegal instruction" errors from invalid kernel launches
    if m < cta_tile_m or n < cta_tile_n:
        return False

    # For cluster shapes > 1, we need enough tiles to fill the cluster
    min_tiles_m = cluster_shape_mn[0]
    min_tiles_n = cluster_shape_mn[1]
    if m < cta_tile_m * min_tiles_m or n < cta_tile_n * min_tiles_n:
        return False

    try:
        gemm = PersistentDenseGemmKernel(
            cutlass.Float32,  # acc_dtype
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            use_tma_store,
            swizzle_size,
            raster_along,
        )
        return gemm.can_implement(
            (m, n, k, batch),
            ab_dtype,
            ab_dtype,  # b_dtype same as a_dtype for FP8
            c_dtype,
            a_major,
            b_major,
            c_major,
        )
    except Exception:
        return False


def _can_implement_config_sm107(
    m: int,
    n: int,
    k: int,
    batch: int,
    config: Tuple,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
) -> bool:
    """Check if an SM107 config can implement the given problem size.

    :param m: M dimension
    :param n: N dimension
    :param k: K dimension
    :param batch: Batch size
    :param config: SM107 config tuple (mma_tiler, mma_inst_shape, cluster_shape_mn,
                   use_2cta_instrs, use_tma_store, swizzle_size, raster_along)
    :param ab_dtype: Input data type (cutlass.Numeric)
    :param c_dtype: Output data type (cutlass.Numeric)
    :param a_major: Major dimension of A ("m" or "k")
    :param b_major: Major dimension of B ("k" or "n")
    :param c_major: Major dimension of C ("m" or "n")
    :return: True if config can implement, False otherwise
    """
    (
        mma_tiler,
        mma_inst_shape,
        cluster_shape_mn,
        use_2cta_instrs,
        use_tma_store,
        swizzle_size,
        raster_along,
    ) = config

    # SM107 FP8 MMA instruction K-mode extent must be 32 or 64 (not 128)
    # This is a hardware constraint of the SM107MmaFP8Op
    mma_inst_k = mma_inst_shape[2]
    if ab_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2) and mma_inst_k not in (
        32,
        64,
    ):
        return False

    # SM107 2-CTA instructions require M=256 in mma_tiler (M=128 causes illegal instruction)
    # This is a hardware constraint specific to SM107's 2-CTA MMA operations
    mma_tiler_m = mma_tiler[0]
    if use_2cta_instrs and mma_tiler_m < 256:
        return False

    # Additional minimum size check: problem must be at least as large as CTA tile
    # For SM107, mma_tiler is (M, N, K) tuple
    cta_tile_m = mma_tiler[0] // (2 if use_2cta_instrs else 1)
    cta_tile_n = mma_tiler[1]

    # Problem size must be able to fill at least one CTA tile
    if m < cta_tile_m or n < cta_tile_n:
        return False

    # For cluster shapes > 1, we need enough tiles to fill the cluster
    min_tiles_m = cluster_shape_mn[0]
    min_tiles_n = cluster_shape_mn[1]
    if m < cta_tile_m * min_tiles_m or n < cta_tile_n * min_tiles_n:
        return False

    try:
        gemm = _sm107_gemm_kernel_cls()(
            cutlass.Float32,  # acc_dtype
            use_2cta_instrs,
            mma_tiler,
            mma_inst_shape,
            cluster_shape_mn,
            use_tma_store,
            swizzle_size,
            raster_along,
        )
        return gemm.can_implement(
            (m, n, k, batch),
            ab_dtype,
            ab_dtype,  # b_dtype same as a_dtype for FP8
            c_dtype,
            a_major,
            b_major,
            c_major,
        )
    except Exception:
        return False


def get_valid_sm100_configs(
    m: int,
    n: int,
    k: int,
    batch: int,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
) -> Tuple[int, ...]:
    """Get indices of valid SM100 configs for the given problem.

    :return: Tuple of valid config indices into SM100_AUTOTUNE_CONFIGS
    """
    valid_indices = []
    for idx, config in enumerate(SM100_AUTOTUNE_CONFIGS):
        if _can_implement_config_sm100(
            m, n, k, batch, config, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            valid_indices.append(idx)
    return tuple(valid_indices)


def get_valid_sm107_configs(
    m: int,
    n: int,
    k: int,
    batch: int,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
) -> Tuple[int, ...]:
    """Get indices of valid SM107 configs for the given problem.

    :return: Tuple of valid config indices into SM107_AUTOTUNE_CONFIGS
    """
    valid_indices = []
    for idx, config in enumerate(SM107_AUTOTUNE_CONFIGS):
        if _can_implement_config_sm107(
            m, n, k, batch, config, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            valid_indices.append(idx)
    return tuple(valid_indices)


# =============================================================================
# Unified BMM Entry Point
# =============================================================================


# Default kernel configurations for each architecture
_DEFAULT_SM100_CONFIG = (
    (128, 128),
    (2, 1),
    True,
    True,
    1,
    "m",
)  # Same as SM100_AUTOTUNE_CONFIGS[0]
_DEFAULT_SM107_CONFIG = (
    (256, 256, 128),
    (256, 256, 64),
    (2, 1),
    True,
    True,
    1,
    "m",
)  # Same as SM107_AUTOTUNE_CONFIGS[0]


def bmm_fp8_cute_dsl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    config_index: Optional[int] = None,
    arch: Optional[Literal["sm100", "sm107"]] = None,
) -> torch.Tensor:
    """Batched matrix multiplication with FP8 inputs using CuTe-DSL backend.

    Computes C = (A @ B) * a_scale * b_scale, where A and B are FP8 tensors.
    The scale multiplication is fused into the kernel's epilogue for optimal performance.

    :param a: Input tensor A of shape (batch, m, k) in FP8 format
    :param b: Input tensor B of shape (batch, k, n) in FP8 format
    :param a_scale: Scale factor for A (scalar or per-tensor)
    :param b_scale: Scale factor for B (scalar or per-tensor)
    :param dtype: Output data type (torch.bfloat16, torch.float16, or torch.float32)
    :param out: Optional pre-allocated output tensor. If None, a new tensor is allocated.
    :param config_index: Optional config index for autotuning. If None, uses default config.
                         Index into SM100_AUTOTUNE_CONFIGS or SM107_AUTOTUNE_CONFIGS.
    :param arch: Optional architecture override ("sm100" or "sm107"). If None, auto-detected
                 from the device compute capability.
    :return: Output tensor C of shape (batch, m, n)
    """
    # Validate inputs
    assert a.dim() == 3, f"Expected 3D tensor for A, got {a.dim()}D"
    assert b.dim() == 3, f"Expected 3D tensor for B, got {b.dim()}D"
    assert a.shape[0] == b.shape[0], "Batch dimensions must match"
    assert a.shape[2] == b.shape[1], "Inner dimensions must match for matmul"

    batch, m, k = a.shape
    _, _, n = b.shape

    # Determine output tensor
    if out is None:
        out = torch.empty((batch, m, n), dtype=dtype, device=a.device)
    else:
        assert out.shape == (batch, m, n), (
            f"Output shape mismatch: {out.shape} vs {(batch, m, n)}"
        )
        assert out.dtype == dtype, f"Output dtype mismatch: {out.dtype} vs {dtype}"

    # Auto-detect architecture if not specified
    if arch is None:
        major, minor = get_compute_capability(a.device)
        if major == 10 and minor == 7:
            arch = "sm107"
        elif major >= 10:
            arch = "sm100"
        else:
            raise ValueError(
                f"CuTe-DSL FP8 BMM is only supported on SM100+ (Blackwell/Rubin), got SM{major}{minor}"
            )

    # Map torch dtype to cutlass dtype
    ab_dtype = torch_dtype_to_cutlass(a.dtype)
    c_dtype = torch_dtype_to_cutlass(dtype)
    acc_dtype = cutlass.Float32

    # Detect memory layout from tensor strides
    a_major = _detect_major_dim(a, ("m", "k"))
    b_major = _detect_major_dim(b, ("k", "n"))
    c_major = "n"

    # Compute combined scale as a tensor to fuse into the kernel epilogue
    combined_scale_tensor = (a_scale * b_scale).float().view(1)

    if arch == "sm107":
        # Get SM107 config - use specified index or default
        if config_index is not None:
            config = SM107_AUTOTUNE_CONFIGS[config_index]
        else:
            config = _DEFAULT_SM107_CONFIG
        (
            mma_tiler,
            mma_inst_shape,
            cluster_shape_mn,
            use_2cta_instrs,
            use_tma_store,
            swizzle_size,
            raster_along,
        ) = config

        _compile_and_run_bmm_sm107(
            a_tensor=a,
            b_tensor=b,
            c_tensor=out,
            ab_dtype=ab_dtype,
            c_dtype=c_dtype,
            acc_dtype=acc_dtype,
            a_major=a_major,
            b_major=b_major,
            c_major=c_major,
            mma_tiler=mma_tiler,
            mma_inst_shape=mma_inst_shape,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
            use_tma_store=use_tma_store,
            swizzle_size=swizzle_size,
            raster_along=raster_along,
            output_scale_tensor=combined_scale_tensor,
        )
    else:
        # SM100 (Blackwell) - default for sm100 and other SM10x variants
        if config_index is not None:
            config = SM100_AUTOTUNE_CONFIGS[config_index]
        else:
            config = _DEFAULT_SM100_CONFIG
        (
            mma_tiler_mn,
            cluster_shape_mn,
            use_2cta_instrs,
            use_tma_store,
            swizzle_size,
            raster_along,
        ) = config

        _compile_and_run_bmm_sm100(
            a_tensor=a,
            b_tensor=b,
            c_tensor=out,
            ab_dtype=ab_dtype,
            c_dtype=c_dtype,
            acc_dtype=acc_dtype,
            a_major=a_major,
            b_major=b_major,
            c_major=c_major,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_2cta_instrs=use_2cta_instrs,
            use_tma_store=use_tma_store,
            swizzle_size=swizzle_size,
            raster_along=raster_along,
            output_scale_tensor=combined_scale_tensor,
        )

    return out


def cute_bmm_fp8_can_implement(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    dtype: torch.dtype,
) -> bool:
    """Check if the CuTe-DSL FP8 BMM kernel can handle the given inputs.

    :param a: Input tensor A
    :param b: Input tensor B
    :param a_scale: Scale factor for A
    :param b_scale: Scale factor for B
    :param dtype: Output data type
    :return: True if kernel can handle inputs, False otherwise
    """
    # Check dimensions
    if a.dim() != 3 or b.dim() != 3:
        return False

    # Check batch and dimension compatibility
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        return False

    # Check dtype
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return False
    if b.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return False

    # Check device compute capability
    try:
        major, minor = get_compute_capability(a.device)
        if major < 10:
            return False
    except Exception:
        return False

    # Check alignment requirements (16-byte alignment for TMA)
    batch, m, k = a.shape
    _, _, n = b.shape

    # For FP8, 16-byte alignment means 16 elements
    if m % 16 != 0 or n % 16 != 0 or k % 16 != 0:
        return False

    return True


__all__ = [
    # Main entry points
    "bmm_fp8_cute_dsl",
    "cute_bmm_fp8_can_implement",
    # Autotune configuration spaces
    "SM100_AUTOTUNE_CONFIGS",
    "SM107_AUTOTUNE_CONFIGS",
    # Configuration validation helpers
    "get_valid_sm100_configs",
    "get_valid_sm107_configs",
]
