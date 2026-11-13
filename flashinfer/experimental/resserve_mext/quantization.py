# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResServe pack16 allocation and FlashInfer JIT dispatch."""

import functools

import torch

from ...jit import env as jit_env
from ...jit.core import current_compilation_context, gen_jit_spec


def gen_resserve_mext_quantize_module():
    return gen_jit_spec(
        "resserve_mext_quantize",
        [jit_env.FLASHINFER_CSRC_DIR / "resserve_mext_quantize.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        ),
    )


@functools.cache
def _module():
    return gen_resserve_mext_quantize_module().build_and_load()


def quantize(input, global_scale):
    m, k = input.shape
    packed = torch.empty((2 * m, k // 2), dtype=torch.uint8, device=input.device)
    scales = torch.zeros(
        ((2 * m + 127) // 128) * ((k + 63) // 64) * 512,
        dtype=torch.uint8,
        device=input.device,
    )
    _module().quantize(input, global_scale, packed, scales)
    return packed, scales
