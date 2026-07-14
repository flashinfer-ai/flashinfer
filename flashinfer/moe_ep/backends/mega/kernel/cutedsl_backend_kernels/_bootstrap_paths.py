# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Put ``kernel/cutedsl_backend_kernels`` on ``sys.path`` for legacy imports.

Called automatically when
:mod:`flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels` is imported
so kernel modules (``src``, ``common``, ``moe_nvfp4_swapab``, ``moe_mxfp8_glu``)
resolve without a separate editable install.
"""

from __future__ import annotations

import os
import sys

_BOOTSTRAPPED = False


def bootstrap_paths() -> None:
    """Idempotently prepend this package directory to ``sys.path``."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    lib_dir = os.path.dirname(os.path.abspath(__file__))
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    _BOOTSTRAPPED = True
