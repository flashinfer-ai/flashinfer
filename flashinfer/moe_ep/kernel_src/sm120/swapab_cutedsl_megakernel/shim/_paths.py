# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Put ``kernel_src/sm120/swapab_cutedsl_megakernel/src`` on ``sys.path`` so kernel packages resolve.

This is shim glue, not kernel-team code, so it lives in ``shim/`` (never inside
``src/`` — that directory is a *verbatim* drop from the kernel team).  It adds
the sibling ``src/`` directory to ``sys.path`` so the raw kernel packages
(``common``, ``moe_sm120_mxfp8_swapab``, ``moe_mxfp8_glu``, ``moe_nvfp4_swapab``,
and the inner ``src``) import as top-level modules without a separate editable
install.

The SM120 drop is a fork of the SM100 kernel repo, so all three vendored trees
(SM100 ``cutedsl_megamoe``, SM90 ``pull_style_cutedsl_megakernel``, and this
one) expose the SAME top-level module names (``common``, ``src``, …).  Only one
tree can be active per process; ``bootstrap_paths`` raises if a sibling tree's
modules are already imported.  That is fine in practice — a process runs on one
GPU architecture, never several.

Called by :mod:`..shim` at import (and re-exported by the package ``__init__``)
before any shim module touches a kernel package.
"""

from __future__ import annotations

import os
import sys

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

    # this file: .../swapab_cutedsl_megakernel/shim/_paths.py
    # -> sibling .../swapab_cutedsl_megakernel/src
    shim_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(shim_dir), "src")

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
