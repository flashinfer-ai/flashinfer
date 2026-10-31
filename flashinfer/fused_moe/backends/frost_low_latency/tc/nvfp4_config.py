# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared admission for packed NVFP4 grouped MMA."""

from ....api import QuantFormat, SwiGLU


def compiler_cuda_major():
    """CUDA used to build the loaded DSL compiler, independent of Torch/nvcc."""
    import cutlass

    return getattr(getattr(cutlass, "CUDA_VERSION", None), "major", None)


def grouped_nvfp4_tactics(config, capability, shape):
    if (
        config.quant.weight is not QuantFormat.NVFP4
        or config.quant.activation not in (QuantFormat.BF16, QuantFormat.FP16)
        or config.quant.output is not config.quant.activation
        or config.activation != SwiGLU()
        or not config.finalize.do_finalize
        or config.routing.top_k != 10
        or config.experts.intermediate_size != 512
        or shape != (8, 2048)
    ):
        return ()
    if capability == (10, 3):
        return (("warp_mma", 4, 2, 256), ("warp_mma", 8, 2, 256))
    if capability != (12, 0):
        return ()
    choices = (
        ("warp_mma", 4, 2, 128),
        ("warp_mma", 8, 4, 128),
        ("warp_mma", 4, 2, 256),
    )
    # The cu12 compiler miscompiles G8/K256, even with DSL 4.8.1 and
    # Torch CUDA 13. Only the loaded cu13 component is qualified for it.
    if compiler_cuda_major() == 13:
        choices += (("warp_mma", 8, 2, 256),)
    return choices
