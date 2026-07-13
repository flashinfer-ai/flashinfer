# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""Trace-time placeholder constructors shared by FMHA TS examples."""

import cutlass
from cutlass.experimental import primitives as prims


def _shape_tuple(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize scalar and tuple shapes for ``cutlass.Array`` construction."""
    if isinstance(shape, tuple):
        return shape
    return (shape,)


def _placeholder_smem_array(
    dtype: type, shape: int | tuple[int, ...] = 1
) -> cutlass.Array | None:
    """Build a typed shared-memory-view placeholder only when an MLIR context exists."""
    try:
        return cutlass.Array(
            cutlass.Int64(0),
            dtype=dtype,
            shape=_shape_tuple(shape),
            addrspace=3,
        )
    except (RuntimeError, ValueError):
        return None


def _placeholder_local_array(
    dtype: type, shape: int | tuple[int, ...] = 1, alignment: int | None = None
) -> cutlass.Array | None:
    """Build a typed local-memory placeholder only when an MLIR context exists."""
    try:
        if alignment is None:
            return cutlass.Array(dtype, shape, space=cutlass.AddressSpace.rmem)
        return cutlass.Array(
            dtype, shape, space=cutlass.AddressSpace.rmem, alignment=alignment
        )
    except (RuntimeError, ValueError):
        return None


def _placeholder_tmem_ptr() -> cutlass.Array | None:
    """Build a typed tensor-memory pointer placeholder only when an MLIR context exists."""
    try:
        return prims.make_tmem_ptr(cutlass.Int32(0), cutlass.Int8)
    except RuntimeError:
        return None
