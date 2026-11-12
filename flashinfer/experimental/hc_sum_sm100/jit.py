# SPDX-License-Identifier: Apache-2.0
"""Shape-specialized HC statistics/Apply through FlashInfer's NVCC JIT."""

import functools
from pathlib import Path

import torch


def require_sources():
    source = Path(__file__).parent / "csrc"
    if not all((source / name).is_file() for name in ("binding.cu", "kernels.cuh")):
        raise RuntimeError(
            "HC sum-state CUDA sources are absent from this FlashInfer installation. "
            "Install a wheel containing the HC JIT sources (nightlies: "
            "https://flashinfer.ai/whl), or install FlashInfer from source."
        )
    return source


@functools.cache
def module(hidden_size, dtype):
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    if hidden_size <= 0 or hidden_size % 512 or hidden_size > 16384:
        raise ValueError("invalid HC hidden size")
    types = {torch.bfloat16: "nv_bfloat16", torch.float16: "nv_half"}
    if dtype not in types:
        raise ValueError("HC supports BF16 and FP16")
    source = require_sources()
    flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )
    return gen_jit_spec(
        f"hc_sum_sm100_h{hidden_size}_{types[dtype]}",
        [source / "binding.cu"],
        extra_cuda_cflags=[
            *flags,
            f"-DHC_HIDDEN={hidden_size}",
            f"-DHC_DTYPE={types[dtype]}",
        ],
        extra_include_paths=[source],
        # The source HC kernels use ordinary FP32 FMA, without FTZ. Preserve
        # BF16 subnormals rather than inheriting the generic FI fast-math default.
        use_fast_math=False,
    ).build_and_load()
