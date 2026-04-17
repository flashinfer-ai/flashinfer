# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility shims for cutlass-dsl version differences.

Centralizes version-dependent API lookups so kernel and role files don't
each carry their own copies.
"""

import cutlass.cute as cute


# setmaxregister_{decrease,increase} added in cutlass-dsl 4.4;
# older versions only have the deprecated warpgroup_reg_{dealloc,alloc}.
setmaxregister_decrease = getattr(
    cute.arch,
    "setmaxregister_decrease",
    getattr(cute.arch, "warpgroup_reg_dealloc", None),
)

setmaxregister_increase = getattr(
    cute.arch,
    "setmaxregister_increase",
    getattr(cute.arch, "warpgroup_reg_alloc", None),
)

# get_max_tmem_alloc_cols added in cutlass-dsl 4.4;
# older versions don't have it.
_TMEM_MAX_ALLOC_COLUMNS_MAP = {"sm_100": 512, "sm_103": 512, "sm_120": 512}


def get_max_tmem_alloc_cols(compute_capability: str) -> int:
    if hasattr(cute.arch, "get_max_tmem_alloc_cols"):
        return cute.arch.get_max_tmem_alloc_cols(compute_capability)
    if compute_capability not in _TMEM_MAX_ALLOC_COLUMNS_MAP:
        raise ValueError(f"Unsupported compute capability: {compute_capability}")
    return _TMEM_MAX_ALLOC_COLUMNS_MAP[compute_capability]


def make_register_tensor(layout_or_shape, dtype):
    """Compat wrapper for register-memory tensor allocation."""
    if hasattr(cute, "make_rmem_tensor"):
        return cute.make_rmem_tensor(layout_or_shape, dtype)
    return cute.make_fragment(layout_or_shape, dtype)


def fence_proxy_async_shared_cta() -> None:
    """Compat wrapper for the async.shared CTA fence_proxy call shape."""
    if hasattr(cute.arch, "ProxyKind") and hasattr(cute.arch, "SharedSpace"):
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        return
    cute.arch.fence_proxy("async.shared", space="cta")


def exp2_fast(x):
    """Compat wrapper for the fast approximate base-2 exponential."""
    if hasattr(cute.math, "exp2"):
        return cute.math.exp2(x, fastmath=True)
    return cute.arch.exp2(x)
