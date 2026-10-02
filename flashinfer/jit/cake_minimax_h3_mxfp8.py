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

"""JIT loader and AOT inventory for the MiniMax-H3 MXFP8 pre-attention stages.

Two generated device sources (``csrc/cake_minimax_h3_mxfp8_pre_attention``)
implement the RMSNorm + AdaLN + MXFP8 quantize stage and the Q/K RMSNorm + RoPE
+ destination-major pack stage. Both are compiled for the exact compute
capability of the device (10.0 or 10.3) with the token count ``M`` (and, for
the pack stage, the partition count ``P``) fixed at compile time through the
shared launcher header; the public API therefore accepts exactly the token
counts listed in :data:`MINIMAX_H3_MXFP8_SHAPES`.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags
from .utils import write_if_different

MiniMaxH3Mxfp8Target = Literal["sm100a", "sm103a"]
MiniMaxH3Mxfp8Stage = Literal[
    "norm_adaln_mxfp8_quantize",
    "qk_rope_destination_mxfp8_pack",
]

MINIMAX_H3_HEADS = 56
MINIMAX_H3_KINDS = 3
MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_SCALE_BLOCK = 32
MINIMAX_H3_SCALE_TILE_ROWS = 128

# (M, P): the exact token counts and destination partitions with a compiled route.
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

_TARGET_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}
_STAGE_INDEX = {"norm_adaln_mxfp8_quantize": 1, "qk_rope_destination_mxfp8_pack": 2}
_DEVICE_SOURCES = (
    "cake_minimax_h3_mxfp8_norm_adaln_quantize_device.cu",
    "cake_minimax_h3_mxfp8_qk_rope_destination_pack_device.cu",
    "cake_minimax_h3_mxfp8_pre_attention_binding.cuh",
)


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_minimax_h3_mxfp8_pre_attention"
    checkout = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "cake_minimax_h3_mxfp8_pre_attention"
    )
    for candidate in (installed, checkout):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "MiniMax-H3 MXFP8 CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def minimax_h3_mxfp8_target(device: torch.device) -> MiniMaxH3Mxfp8Target:
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability == (10, 0):
        return "sm100a"
    if capability == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "MiniMax-H3 MXFP8 pre-attention requires exact compute capability "
        f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
    )


def minimax_h3_mxfp8_require_route(M: int, P: int) -> None:
    """Raise unless ``(M, P)`` is one of the compiled exact routes."""
    if (M, P) not in MINIMAX_H3_MXFP8_SHAPES:
        raise RuntimeError(f"no exact MiniMax-H3 MXFP8 route for M={M}, P={P}")


def _schedule_constants(M: int, P: int, stage: MiniMaxH3Mxfp8Stage) -> dict[str, int]:
    constants = {"STAGE": _STAGE_INDEX[stage], "M": M}
    if stage == "qk_rope_destination_mxfp8_pack":
        heads_per_destination = MINIMAX_H3_HEADS // P
        rows_per_destination = M * heads_per_destination * MINIMAX_H3_KINDS
        constants.update(
            P=P,
            HEADS_PER_DESTINATION=heads_per_destination,
            ROWS_PER_DESTINATION=rows_per_destination,
            SCALE_STRIDE=_round_up(rows_per_destination, MINIMAX_H3_SCALE_TILE_ROWS)
            * (MINIMAX_H3_HEAD_DIM // MINIMAX_H3_SCALE_BLOCK),
        )
    return constants


def _binding_source(constants: dict[str, int]) -> str:
    defines = "".join(
        f"#define CAKE_MINIMAX_H3_MXFP8_{name} {value}\n" for name, value in constants.items()
    )
    return (
        "/*\n"
        " * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.\n"
        " * Licensed under the Apache License, Version 2.0.\n"
        " */\n"
        f"{defines}"
        '#include "cake_minimax_h3_mxfp8_pre_attention_binding.cuh"\n'
    )


@functools.cache
def gen_minimax_h3_mxfp8_stage_module(
    target: MiniMaxH3Mxfp8Target,
    M: int,
    P: int,
    stage: MiniMaxH3Mxfp8Stage,
) -> JitSpec:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported MiniMax-H3 MXFP8 target: {target}")
    if stage not in _STAGE_INDEX:
        raise ValueError(f"unsupported MiniMax-H3 MXFP8 stage: {stage}")
    minimax_h3_mxfp8_require_route(M, P)
    csrc_dir = _get_csrc_dir()
    for required in _DEVICE_SOURCES:
        if not (csrc_dir / required).is_file():
            raise FileNotFoundError(
                f"MiniMax-H3 MXFP8 source was not installed: {csrc_dir / required}"
            )
    constants = _schedule_constants(M, P, stage)
    # The quantize stage depends on M only; every P shares its module.
    instance = f"m{M}" if stage == "norm_adaln_mxfp8_quantize" else f"m{M}_p{P}"
    uri = f"cake_minimax_h3_mxfp8_{stage}_{target}_{instance}"
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / f"{uri}.cu"
    write_if_different(binding, _binding_source(constants))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[*_TARGET_FLAGS[target], "--use_fast_math"],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
    )
    logger.info("Generated MiniMax-H3 MXFP8 %s JIT spec: %s", stage, spec.name)
    return spec


@functools.cache
def load_minimax_h3_mxfp8_stage_module(
    target: MiniMaxH3Mxfp8Target,
    M: int,
    P: int,
    stage: MiniMaxH3Mxfp8Stage,
):
    return gen_minimax_h3_mxfp8_stage_module(target, M, P, stage).build_and_load()


def load_minimax_h3_mxfp8_route(device: torch.device, M: int, P: int):
    """Return the ``(quantize, pack)`` modules for ``device`` at ``(M, P)``."""
    target = minimax_h3_mxfp8_target(device)
    return (
        load_minimax_h3_mxfp8_stage_module(target, M, P, "norm_adaln_mxfp8_quantize"),
        load_minimax_h3_mxfp8_stage_module(
            target, M, P, "qk_rope_destination_mxfp8_pack"
        ),
    )


def gen_minimax_h3_mxfp8_aot_modules(
    target: MiniMaxH3Mxfp8Target,
) -> tuple[JitSpec, ...]:
    """Return the deduplicated JIT specs of every exact route for one target."""
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported MiniMax-H3 MXFP8 target: {target}")
    specs: dict[str, JitSpec] = {}
    for M, P in MINIMAX_H3_MXFP8_SHAPES:
        for stage in _STAGE_INDEX:
            spec = gen_minimax_h3_mxfp8_stage_module(target, M, P, stage)
            specs.setdefault(spec.name, spec)
    return tuple(specs.values())


__all__ = [
    "MINIMAX_H3_MXFP8_SHAPES",
    "MiniMaxH3Mxfp8Stage",
    "MiniMaxH3Mxfp8Target",
    "gen_minimax_h3_mxfp8_aot_modules",
    "gen_minimax_h3_mxfp8_stage_module",
    "load_minimax_h3_mxfp8_route",
    "load_minimax_h3_mxfp8_stage_module",
    "minimax_h3_mxfp8_require_route",
    "minimax_h3_mxfp8_target",
]
