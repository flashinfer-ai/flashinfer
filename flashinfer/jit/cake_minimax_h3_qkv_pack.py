# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AOT inventory for the exact-architecture MiniMax-H3 QKV quantize-and-pack programs."""

from __future__ import annotations

import importlib
from typing import Literal

from .core import JitSpec

MiniMaxH3QkvPackTarget = Literal["sm100a", "sm103a"]

# Destination partition counts and output formats of the generated pack
# programs. The token count M is a runtime parameter, so the inventory is one
# program per (P, format): eight programs per exact target.
MINIMAX_H3_QKV_PACK_PARTITIONS = (1, 2, 4, 8)
MINIMAX_H3_QKV_PACK_FORMATS = ("nvfp4", "mxfp8")


def gen_minimax_h3_qkv_pack_aot_modules(
    target: MiniMaxH3QkvPackTarget,
) -> tuple[JitSpec, ...]:
    """Return the deduplicated JIT specs for one exact physical target."""

    if target not in ("sm100a", "sm103a"):
        raise ValueError(f"unsupported MiniMax-H3 QKV pack target: {target}")
    module = importlib.import_module(
        f".cake_minimax_h3_qkv_quantize_pack_{target}", __package__
    )
    specs: dict[str, JitSpec] = {}
    for fmt in MINIMAX_H3_QKV_PACK_FORMATS:
        for P in MINIMAX_H3_QKV_PACK_PARTITIONS:
            route = module.minimax_h3_qkv_pack_route_record(P, fmt)
            if (
                route.get("target") != target
                or route.get("P") != P
                or route.get("format") != fmt
            ):
                raise RuntimeError(
                    f"MiniMax-H3 QKV pack route identity mismatch for {target}:{P}:{fmt}"
                )
            spec = module.gen_minimax_h3_qkv_pack_module(P, fmt)
            specs.setdefault(spec.name, spec)
    return tuple(specs.values())


__all__ = [
    "MINIMAX_H3_QKV_PACK_FORMATS",
    "MINIMAX_H3_QKV_PACK_PARTITIONS",
    "MiniMaxH3QkvPackTarget",
    "gen_minimax_h3_qkv_pack_aot_modules",
]
