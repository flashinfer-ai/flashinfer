# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Minimal torchrun smoke for the NVFP4 MegaMoE thin API."""

from __future__ import annotations

from .nvfp4 import _main

if __name__ == "__main__":
    _main()
