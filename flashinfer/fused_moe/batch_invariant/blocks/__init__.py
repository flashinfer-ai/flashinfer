# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped (expert-segmented) GEMM as a scheduler placement over the dense GEMM blocks: group scheduler, per-tile
TMA coordinates, group-masked epilogue with the plain, the gated (SwiGLU) or the ungated (ReLU^2) chain, kernel
composition and compile wrapper."""
