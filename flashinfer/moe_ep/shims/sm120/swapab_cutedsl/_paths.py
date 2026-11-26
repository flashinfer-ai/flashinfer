# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Load this pinned legacy export, rejecting conflicting raw packages."""

from __future__ import annotations

import os
import sys

from flashinfer.moe_ep.shims._sources import source_root

_BOOTSTRAPPED = False

# These packages are imported without a namespace by the SM120 swap-AB
# sources. Reject copies already loaded from a different source tree.
_SENTINEL_MODULES = ("common", "src", "moe_nvfp4_swapab", "moe_mxfp8_glu")


def bootstrap_paths() -> None:
    """Idempotently prepend the SM120 swap-AB source directory to ``sys.path``."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    src_dir = str(source_root("flashinfer", "sm120_swapab"))

    for name in _SENTINEL_MODULES:
        mod = sys.modules.get(name)
        mod_file = getattr(mod, "__file__", None) if mod is not None else None
        if mod_file is not None and not os.path.abspath(mod_file).startswith(
            src_dir + os.sep
        ):
            raise RuntimeError(
                f"kernel module {name!r} is already imported from "
                f"{mod_file!r}; SM120 swap-AB requires its own copy. "
                "Run the workloads that load these packages in separate processes."
            )

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    _BOOTSTRAPPED = True
