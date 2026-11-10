# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Put ``kernel_src/sm100/cutedsl_megamoe/src`` on ``sys.path`` so kernel packages resolve.

This is shim glue, not kernel-team code, so it lives in ``shim/`` (never inside
``src/`` — that directory is a *verbatim* drop from the kernel team).  It adds
the sibling ``src/`` directory to ``sys.path`` so the raw kernel packages
(``common``, ``moe_nvfp4_swapab``, ``moe_mxfp8_glu``, and the inner ``src``)
import as top-level modules without a separate editable install.

The SM90 tree (``kernel_src/sm90/pull_style_cutedsl_megakernel``) is a fork of this kernel
repo and exposes the SAME top-level module names; only one tree can be active
per process (a process runs on either Blackwell or Hopper, never both), so
``bootstrap_paths`` raises if the sibling tree's modules are already imported.

Called by :mod:`..shim` at import (and re-exported by the package ``__init__``)
before any shim module touches a kernel package.
"""

from __future__ import annotations

import os
import sys

_BOOTSTRAPPED = False

# Any of these already imported from a DIFFERENT src tree means the sibling
# (SM90) backend owns this process's kernel modules.
_SENTINEL_MODULES = ("common", "src", "moe_nvfp4_swapab")


def bootstrap_paths() -> None:
    """Idempotently prepend the vendored ``src/`` directory to ``sys.path``."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # this file: .../cutedsl_megamoe/shim/_paths.py -> sibling .../cutedsl_megamoe/src
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
                f"{mod_file!r}; the SM100 and SM90 cutedsl_megamoe trees share "
                "top-level module names and cannot be active in one process. "
                "Use a separate process for the other architecture's backend."
            )

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    _BOOTSTRAPPED = True
