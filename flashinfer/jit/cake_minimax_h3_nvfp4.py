# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AOT inventory of the Cake MiniMax-H3 NVFP4 pre-attention programs."""

from __future__ import annotations

from .cake_minimax_h3_nvfp4_pre_attention import (
    MINIMAX_H3_NVFP4_STAGES,
    MiniMaxH3Nvfp4Target,
    gen_minimax_h3_nvfp4_stage_module,
)
from .core import JitSpec

# Destination partition counts of the prepared operation. The token count M
# and the destination geometry are runtime launch parameters, so every
# partition count of one target uses the same two programs.
MINIMAX_H3_NVFP4_PARTITIONS = (1, 2, 4, 8)


def gen_minimax_h3_nvfp4_aot_modules(
    target: MiniMaxH3Nvfp4Target,
) -> tuple[JitSpec, ...]:
    """Return the deduplicated JIT specs for one exact physical target."""

    if target not in ("sm100a", "sm103a"):
        raise ValueError(f"unsupported MiniMax-H3 NVFP4 target: {target}")
    specs: dict[str, JitSpec] = {}
    for stage in MINIMAX_H3_NVFP4_STAGES:
        spec = gen_minimax_h3_nvfp4_stage_module(target, stage)
        specs.setdefault(spec.name, spec)
    return tuple(specs.values())


__all__ = [
    "MINIMAX_H3_NVFP4_PARTITIONS",
    "MiniMaxH3Nvfp4Target",
    "gen_minimax_h3_nvfp4_aot_modules",
]
