# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile (cuda.tile Python) kernel implementations for FlashInfer.

This subpackage provides cuTile-based kernels that mirror selected FlashInfer
public APIs, using the public ``cuda.tile`` Python API directly (no external
DSL middleware). See ``flashinfer.cutile.gemm`` for the GEMM family.

Kernels in this subpackage are ported from NVIDIA TileGym
(https://github.com/NVIDIA/TileGym), which is the upstream home of the
operator implementations. The ports keep the kernel body verbatim and strip
TileGym-internal decorators (``@register_impl``) and autotune helpers in
favor of the equivalent public ``cuda.tile`` APIs.
"""
