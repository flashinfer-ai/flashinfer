# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy public access to the shared Blackwell/Rubin work-ID protocol."""

from importlib import import_module

from ._paths import bootstrap_paths

_HELPER_NAMES = (
    "AtomicCounterWorkIdState",
    "FixedGroupMixedCgaAtomicCounterWorkIdState",
    "claim_work_id",
    "initialize_fixed_group_mixed_cga_work_id_state",
)
__all__ = list(_HELPER_NAMES)


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    bootstrap_paths()
    module = import_module(
        "blackwell_genphase_sources.kernel_src.schedulers.work_id_claim"
    )
    return getattr(module, name)
