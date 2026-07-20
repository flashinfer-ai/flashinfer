# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/w4a8/__init__.py @ 731ba9da (2026-06-30) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""W4A8-specific prepared-weight utilities for the unified MoE kernel."""

from .weights import repack_w4a8_weights

__all__ = ["repack_w4a8_weights"]
