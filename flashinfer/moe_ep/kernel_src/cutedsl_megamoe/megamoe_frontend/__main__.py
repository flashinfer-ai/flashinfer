# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Entry point for ``python -m flashinfer.moe_ep.kernel_src.cutedsl_megamoe.megamoe_frontend``."""

from __future__ import annotations

from ..correctness import main

if __name__ == "__main__":
    raise SystemExit(main())
