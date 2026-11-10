# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility import for the shared BF16 input stager."""

from .....core.bf16_stage import fused_bf16_stage

__all__ = ["fused_bf16_stage"]
