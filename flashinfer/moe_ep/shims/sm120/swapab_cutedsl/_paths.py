# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Load this pinned legacy export, rejecting conflicting raw packages."""

from __future__ import annotations

import os
import sys

from flashinfer.moe_ep.shims._sources import source_root

_BOOTSTRAPPED = False

# Any of these already imported from a DIFFERENT src tree means a sibling
# backend owns this process's kernel modules.  ``common`` / ``src`` catch the
# SM90 tree; ``moe_nvfp4_swapab`` / ``moe_mxfp8_glu`` additionally catch the
# SM100 tree (which ships both packages at its own drop revision).
_SENTINEL_MODULES = ("common", "src", "moe_nvfp4_swapab", "moe_mxfp8_glu")


def bootstrap_paths() -> None:
    """Idempotently prepend the vendored ``src/`` directory to ``sys.path``."""
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
                f"{mod_file!r}; the SM100 cutedsl_megamoe, SM90 "
                "pull_style_cutedsl_megakernel, and SM120 "
                "swapab_cutedsl_megakernel trees share top-level module names "
                "and cannot be active in one process. Use a separate process "
                "for the other architecture's backend."
            )

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    _BOOTSTRAPPED = True
