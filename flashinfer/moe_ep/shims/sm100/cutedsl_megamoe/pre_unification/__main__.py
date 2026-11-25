# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Entry point so ``python -m flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification``
runs the correctness smoke test documented in ``correctness.py``."""

import sys

from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.correctness import (
    main,
)

if __name__ == "__main__":
    sys.exit(main())
