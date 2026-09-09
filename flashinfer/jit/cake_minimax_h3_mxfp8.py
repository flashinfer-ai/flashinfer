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

"""AOT inventory for exact-architecture MiniMax-H3 generated stages."""

from __future__ import annotations

import importlib
from typing import Literal

from .core import JitSpec

MiniMaxH3Mxfp8Target = Literal["sm100a", "sm103a"]

MINIMAX_H3_MXFP8_SHAPES = (
    (33472, 1),
    (16736, 2),
    (8368, 4),
    (4184, 8),
    (38592, 1),
    (19296, 2),
    (9648, 4),
    (4824, 8),
    (48768, 1),
    (24384, 2),
    (12192, 4),
    (6096, 8),
    (58944, 1),
    (29472, 2),
    (14736, 4),
    (7368, 8),
    (74240, 1),
    (37120, 2),
    (18560, 4),
    (9280, 8),
    (109952, 1),
    (54976, 2),
    (27488, 4),
    (13744, 8),
    (38528, 1),
    (38656, 1),
    (19264, 2),
    (19328, 2),
    (9632, 4),
    (9664, 4),
    (4816, 8),
    (4832, 8),
    (38591, 1),
    (38593, 1),
    (19295, 2),
    (19297, 2),
    (9647, 4),
    (9649, 4),
    (4823, 8),
    (4825, 8),
    (1, 8),
    (127, 8),
    (128, 8),
    (129, 8),
)

_STAGES = (
    "norm_adaln_mxfp8_quantize",
    "qk_rope_destination_mxfp8_pack",
)


def gen_minimax_h3_mxfp8_aot_modules(
    target: MiniMaxH3Mxfp8Target,
) -> tuple[JitSpec, ...]:
    """Return the deduplicated JIT specs for one exact physical target."""

    if target not in ("sm100a", "sm103a"):
        raise ValueError(f"unsupported MiniMax-H3 MXFP8 target: {target}")
    module = importlib.import_module(
        f".cake_minimax_h3_mxfp8_pre_attention_{target}", __package__
    )
    specs: dict[str, JitSpec] = {}
    for M, P in MINIMAX_H3_MXFP8_SHAPES:
        route = module.minimax_h3_mxfp8_route_record(M, P)
        if route.get("target") != target or route.get("M") != M or route.get("P") != P:
            raise RuntimeError(
                f"MiniMax-H3 MXFP8 route identity mismatch for {target}:{M}:{P}"
            )
        for stage in _STAGES:
            spec = module.gen_minimax_h3_mxfp8_stage_module(M, P, stage)
            specs.setdefault(spec.name, spec)
    return tuple(specs.values())


__all__ = [
    "MINIMAX_H3_MXFP8_SHAPES",
    "MiniMaxH3Mxfp8Target",
    "gen_minimax_h3_mxfp8_aot_modules",
]
