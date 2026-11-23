# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Load the namespaced provider package without changing Python's search path."""

from importlib import import_module


def bootstrap_paths() -> None:
    import_module("flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification")
