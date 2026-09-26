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

MiniMaxH3Nvfp4Target = Literal["sm100a", "sm103a"]

# Destination partition counts of the prepared operation. The token count M
# is a runtime parameter of both stages and the fused QKV GEMM takes the
# destination geometry at launch, so the four partition routes of one target
# deduplicate to two programs: the norm stage and the fused GEMM + pack stage.
MINIMAX_H3_NVFP4_PARTITIONS = (1, 2, 4, 8)

_STAGES = (
    "norm_adaln_nvfp4_quantize",
    "qkv_nvfp4_gemm_fused_pack",
)


def gen_minimax_h3_nvfp4_aot_modules(
    target: MiniMaxH3Nvfp4Target,
) -> tuple[JitSpec, ...]:
    """Return the deduplicated JIT specs for one exact physical target."""

    if target not in ("sm100a", "sm103a"):
        raise ValueError(f"unsupported MiniMax-H3 NVFP4 target: {target}")
    module = importlib.import_module(
        f".cake_minimax_h3_nvfp4_pre_attention_{target}", __package__
    )
    specs: dict[str, JitSpec] = {}
    for P in MINIMAX_H3_NVFP4_PARTITIONS:
        route = module.minimax_h3_nvfp4_route_record(P)
        if route.get("target") != target or route.get("P") != P:
            raise RuntimeError(
                f"MiniMax-H3 NVFP4 route identity mismatch for {target}:{P}"
            )
        for stage in _STAGES:
            spec = module.gen_minimax_h3_nvfp4_stage_module(P, stage)
            specs.setdefault(spec.name, spec)
    return tuple(specs.values())


__all__ = [
    "MINIMAX_H3_NVFP4_PARTITIONS",
    "MiniMaxH3Nvfp4Target",
    "gen_minimax_h3_nvfp4_aot_modules",
]
