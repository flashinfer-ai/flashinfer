# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AOT inventory of the Cake MiniMax-H3 QKV quantize-and-pack programs."""

from __future__ import annotations

from .cake_minimax_h3_qkv_quantize_pack import (
    ROUTES,
    MiniMaxH3QkvPackTarget,
    gen_minimax_h3_qkv_pack_module,
)
from .core import JitSpec

# Destination partition counts and output formats of the generated programs.
# The token count M is a runtime parameter and P is a compile-line definition
# of the route, so the inventory is one build per (P, format) of one source
# per format.
MINIMAX_H3_QKV_PACK_PARTITIONS = (1, 2, 4, 8)
MINIMAX_H3_QKV_PACK_FORMATS = ("nvfp4", "mxfp8")


def gen_minimax_h3_qkv_pack_aot_modules(
    target: MiniMaxH3QkvPackTarget,
) -> tuple[JitSpec, ...]:
    """Return the deduplicated JIT specs for one exact physical target."""

    if target not in ("sm100a", "sm103a"):
        raise ValueError(f"unsupported MiniMax-H3 QKV pack target: {target}")
    specs: dict[str, JitSpec] = {}
    for key in ROUTES:
        fmt, P = key.split(":")
        spec = gen_minimax_h3_qkv_pack_module(target, int(P), fmt)  # type: ignore[arg-type]
        specs.setdefault(spec.name, spec)
    return tuple(specs.values())


__all__ = [
    "MINIMAX_H3_QKV_PACK_FORMATS",
    "MINIMAX_H3_QKV_PACK_PARTITIONS",
    "MiniMaxH3QkvPackTarget",
    "gen_minimax_h3_qkv_pack_aot_modules",
]
