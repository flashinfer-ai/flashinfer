# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Training-mega-specific numeric constants (MMA tiler extents).

These are consumed only by the training GLU/dGLU FC12 kernels, so they live in the
training mega ``helpers`` package rather than the cross-inference/training
``next/sources/helpers/constants.py``.
"""


# MMA tiler GLU FC12 kernels accept along M and N.
SupportedMmaTileM = (128, 256)
SupportedMmaTileN = (128, 256)


__all__ = [
    "SupportedMmaTileM",
    "SupportedMmaTileN",
]
