# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Fixed routing/finalize adapters shared by Frost JIT and AOT builds."""

from pathlib import Path

from .core import gen_jit_spec, sm107a_nvcc_flags, sm120a_nvcc_flags

FROST_DTYPES = ("bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4")


def gen_cudnn_frost_moe_module(dtype: str, arch: str = "sm_107a"):
    if dtype not in FROST_DTYPES:
        raise ValueError(f"Unsupported Frost MoE dtype: {dtype}")
    if arch == "sm_107a":
        flags = sm107a_nvcc_flags
    elif arch == "sm_120a" and dtype == "bf16":
        flags = sm120a_nvcc_flags
    else:
        raise ValueError("cuDNN Frost MoE adapters require SM107a, or SM120a for BF16")
    source = (
        Path(__file__).parents[1]
        / "fused_moe/backends/cudnn_frost/csrc"
        / f"moe_{dtype}.cu"
    )
    return gen_jit_spec(
        f"cudnn_frost_{dtype}_moe_v2_{arch}",
        [source],
        extra_cuda_cflags=flags,
    )
