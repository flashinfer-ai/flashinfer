# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy access to the GenPhase mixed-cluster TMA protocol."""

from ._paths import bootstrap_paths

_HELPER_NAMES = (
    "PipelineTmaUmmaMixedCga",
    "bind_executable_tma_load_fields",
    "select_tma_atom_for_cluster",
)
__all__ = list(_HELPER_NAMES)


def __getattr__(name):
    if name in __all__:
        bootstrap_paths()
        from blackwell_genphase_sources.kernel_src.blackwell import (
            custom_mix_cga_helpers,
        )

        return getattr(custom_mix_cga_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
