# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResInfer pack16 allocation and FlashInfer JIT dispatch."""

import functools

import torch

from ...jit.resinfer import gen_resinfer_mext_quantize_module


@functools.cache
def _module():
    return gen_resinfer_mext_quantize_module().build_and_load()


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
