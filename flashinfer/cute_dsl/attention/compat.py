# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility shims for cutlass-dsl version differences.

Centralizes version-dependent API lookups so kernel and role files don't
each carry their own copies.
"""

import torch

import cutlass.cute as cute

from ...utils import get_compute_capability


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
_TMEM_MAX_ALLOC_COLUMNS_MAP = {
    "sm_100": 512,
    "sm_103": 512,
    "sm_120": 512,
    "sm_121": 512,
}


def get_max_tmem_alloc_cols(compute_capability: str) -> int:
    if hasattr(cute.arch, "get_max_tmem_alloc_cols"):
        return cute.arch.get_max_tmem_alloc_cols(compute_capability)
    if compute_capability not in _TMEM_MAX_ALLOC_COLUMNS_MAP:
        raise ValueError(f"Unsupported compute capability: {compute_capability}")
    return _TMEM_MAX_ALLOC_COLUMNS_MAP[compute_capability]


def get_current_arch() -> str:
    """Return the cutlass-dsl arch string ('sm_100', 'sm_103', ...)
    for the current default CUDA device.

    When the installed DSL lacks the device's own architecture but can
    target it through the family-conditional arch (e.g. sm_107 via
    sm_100f), return the family base ('sm_100') so capacity lookups and
    kernel configuration use family-portable values."""
    major, minor = get_compute_capability(torch.device("cuda"))
    from ..utils import is_cute_dsl_arch_supported

    if not is_cute_dsl_arch_supported(
        major, minor, native_only=True
    ) and is_cute_dsl_arch_supported(major, minor):
        return f"sm_{major}0"
    return f"sm_{major}{minor}"
