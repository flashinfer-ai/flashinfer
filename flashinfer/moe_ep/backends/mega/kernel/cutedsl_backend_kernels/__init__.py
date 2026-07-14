# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CuTeDSL MegaMoE kernel library shipped with FlashInfer ``moe_ep``."""

from __future__ import annotations

from ._bootstrap_paths import bootstrap_paths

bootstrap_paths()

__all__ = ["bootstrap_paths"]
