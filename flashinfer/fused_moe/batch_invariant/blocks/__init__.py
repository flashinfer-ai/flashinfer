# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped (expert-segmented) GEMM as a scheduler placement over the dense GEMM blocks, for the bf16, the fp8
per-tensor and the NVFP4 MoE classes: group scheduler, the operand blocks (the dense formats' ring; the block-scaled formats'
scale-factor rings and padded group tables), per-tile TMA coordinates, the grouped and the swapped body (one source per body,
the operand format and the epilogue chain resolved at trace time), the group-masked epilogue with the plain, the gated
(SwiGLU and the other gated activations) or the ungated (ReLU^2 and the others) chain, the W8A8 chains and the W4A4 chains
(the per-row scale, the static class's fused block quantization), kernel composition and compile wrapper; the NVFP4 classes'
destination-mapped quantizers (:mod:`.quant_nvfp4`)."""
