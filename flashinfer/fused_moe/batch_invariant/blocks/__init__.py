# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped (expert-segmented) GEMM as a scheduler placement over the dense GEMM blocks, for the bf16 and the fp8
per-tensor MoE classes: group scheduler, per-tile TMA coordinates, the grouped and the swapped body (one source per body,
the operand format and the epilogue chain resolved at trace time), the group-masked epilogue with the plain, the gated
(SwiGLU and the other gated activations) or the ungated (ReLU^2 and the others) chain and the W8A8 chains, kernel composition
and compile wrapper."""
