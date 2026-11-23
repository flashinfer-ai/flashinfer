# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared constants for the fused fc1+fc2 MegaMoE path."""

Log2E = 1.4426950408889634
Fp32Max = 3.40282346638528859812e38

Nvfp4BlockSize = 16
Mxfp8BlockSize = 32
Fp8E8M0SfVecSize = 32
Fp8DispatchScaleAtomK = 4 * Fp8E8M0SfVecSize
Fp8GateUpInterleave = 8
Fp8BlockScaleK = 128
# Row-scale block (columns per fp32 scale) of the activation operand the FC1 /
# dfc2 epilogues quantize in-kernel: the forward h (= SwiGLU output, FC2's A
# operand) and the backward grad_y1 (dfc1's A operand).  128 = one scale per K
# tile, promoted once per tile like the 1x128 x / dY operands (a 1x64 block
# needed two planes and two promotions per K tile).  The epilogue forms
# one block from the channels one task tile produces, so the kernels require
# >= 128 output channels per task (non-swap N256, swap-AB M256).
Fp8Fc2ActivationScaleK = 128
# The activation scales ride in each AB stage as 4 fp32 planes per token row
# (one 16-byte TMA row); a K tile consumes Fp8Fc2ActSfPlanesPerKTile of them,
# so one TMA row serves Fp8Fc2ActSfKTilesPerGroup consecutive K tiles.
Fp8ActSfPlanes = 4
Fp8Fc2ActSfPlanesPerKTile = Fp8BlockScaleK // Fp8Fc2ActivationScaleK
Fp8Fc2ActSfKTilesPerGroup = Fp8ActSfPlanes // Fp8Fc2ActSfPlanesPerKTile
Fp8WeightScaleBlockN = 128
Fp8WeightScaleBlockK = 128
SfPaddingBlock = 128
TmaLeadingDimByteAlign = 16

Nvfp4E2M1Max = 6.0
Fp8E4M3FNMax = 448.0
Fp8E5M2Max   = 57344.0

Nvfp4E2M1RcpLimit = 1.0 / Nvfp4E2M1Max
Fp8E4M3RcpLimit   = 1.0 / Fp8E4M3FNMax
Fp8E5M2RcpLimit   = 1.0 / Fp8E5M2Max

SupportedMmaTileM = (64, 128, 256)
SupportedMmaTileN = (64, 128, 256)
