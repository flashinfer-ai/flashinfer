# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""KEXT allocation and pack dispatch from public ResInfer commit 51b9068.

Sources: csrc/nvfp4/quant.py and csrc/nvfp4/cuda/nvfp4_quant_entry.cu.
"""

import functools

import torch

from ...jit.resinfer import gen_resinfer_kext_quantize_module


@functools.cache
def _module():
    return gen_resinfer_kext_quantize_module().build_and_load()


def _pack_mode(m, k, residue_per_8):
    # Preserve the donor's SM100 shape policy, without its global device cache.
    if m <= 128:
        if residue_per_8 == 1 and (k in (256, 768) or (m >= 4 and k in (512, 1024))):
            return 8
        if residue_per_8 == 2 and k in (8192, 12288, 16384):
            return 16
        if residue_per_8 == 4 and k == 16384:
            return 16
    return 16 if residue_per_8 == 1 or m >= 1024 else 8


def quantize(input, global_scale, channel_masks, residue_per_8, elts_per_thread):
    m, k = input.shape
    ext = k + k // 8 * residue_per_8
    packed = torch.empty((m, ext // 2), dtype=torch.uint8, device=input.device)
    scales = torch.zeros(
        ((m + 127) // 128) * ((ext + 63) // 64) * 512,
        dtype=torch.uint8,
        device=input.device,
    )
    mode = elts_per_thread or _pack_mode(m, k, residue_per_8)
    _module().quantize(
        input, global_scale, channel_masks, packed, scales, residue_per_8, mode
    )
    return packed, scales
