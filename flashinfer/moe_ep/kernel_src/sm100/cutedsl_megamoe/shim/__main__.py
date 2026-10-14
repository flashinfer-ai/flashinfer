# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Entry point so ``python -m flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim``
runs the correctness smoke test documented in ``correctness.py``."""

import sys

from .correctness import main

if __name__ == "__main__":
    sys.exit(main())
