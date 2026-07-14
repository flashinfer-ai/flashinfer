# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Entry point for ``python -m cutedsl_megamoe_front_end.megamoe_frontend``."""

from __future__ import annotations

from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
