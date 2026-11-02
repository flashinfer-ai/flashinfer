# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Fixed routing/finalize adapters shared by Frost JIT and AOT builds."""

from pathlib import Path

from .core import gen_jit_spec, sm107a_nvcc_flags, sm120a_nvcc_flags

FROST_DTYPES = ("bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4")
# Every dtype packages SM107a and SM120a sources; each adapter targets one arch.
FROST_ARCH_FLAGS = {"sm_107a": sm107a_nvcc_flags, "sm_120a": sm120a_nvcc_flags}


def gen_cudnn_frost_moe_module(dtype: str, arch: str = "sm_107a"):
    if dtype not in FROST_DTYPES:
        raise ValueError(f"Unsupported Frost MoE dtype: {dtype}")
    flags = FROST_ARCH_FLAGS.get(arch)
    if flags is None:
        raise ValueError("cuDNN Frost MoE adapters require SM107a or SM120a")
    source = (
        Path(__file__).parents[1]
        / "fused_moe/backends/cudnn_frost/csrc"
        / f"moe_{dtype}.cu"
    )
    return gen_jit_spec(
        f"cudnn_frost_{dtype}_moe_{arch}",
        [source],
        extra_cuda_cflags=flags,
    )
