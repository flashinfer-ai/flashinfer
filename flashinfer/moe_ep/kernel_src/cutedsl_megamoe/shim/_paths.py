# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Put ``kernel_src/cutedsl_megamoe/src`` on ``sys.path`` so kernel packages resolve.

This is shim glue, not kernel-team code, so it lives in ``shim/`` (never inside
``src/`` — that directory is a *verbatim* drop from the kernel team).  It adds
the sibling ``src/`` directory to ``sys.path`` so the raw kernel packages
(``common``, ``moe_nvfp4_swapab``, ``moe_mxfp8_glu``, and the inner ``src``)
import as top-level modules without a separate editable install.

Called by :mod:`..shim` at import (and re-exported by the package ``__init__``)
before any shim module touches a kernel package.
"""

from __future__ import annotations

import os
import sys

_BOOTSTRAPPED = False


def bootstrap_paths() -> None:
    """Idempotently prepend the vendored ``src/`` directory to ``sys.path``."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # this file: .../cutedsl_megamoe/shim/_paths.py -> sibling .../cutedsl_megamoe/src
    shim_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(shim_dir), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    _BOOTSTRAPPED = True
