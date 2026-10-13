# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the Cake MiniMax-H3 QKV quantize-and-pack programs.

One generated program per output format (``nvfp4`` / ``mxfp8``) serves every
destination partition count ``P`` and both exact targets: the token count
``M`` and the derived strides are runtime launch parameters, ``P`` and
``HEADS_PER_DESTINATION = 56 // P`` are compile-line definitions of the route,
and the source is compiled with the exact flag set of the device it runs on.
``MODULES`` and ``ROUTES`` are written by the Cake export; the empty tables are
the source-only placeholder and every route lookup fails until they are filled.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags

MiniMaxH3QkvPackFormat = Literal["nvfp4", "mxfp8"]
MiniMaxH3QkvPackTarget = Literal["sm100a", "sm103a"]

_TARGET_FLAGS: dict[str, list[str]] = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

# program name -> {"sources": [kernel, binding] (relative to the csrc family
# directory), "template", "ffi_entry", "compile_flags", "arg_plan",
# "launch_grid_rule", "launch_block", "cluster", "dynamic_smem_bytes",
# "use_pdl", "closure": {target: sealed closure identity}}.
MODULES: dict[str, dict[str, Any]] = {
    "cake_minimax_h3_qkv_quantize_pack_766995e25961b5cee390": {
        "sources": [
            "cake_minimax_h3_qkv_quantize_pack_766995e25961b5cee390_kernel.cu",
            "cake_minimax_h3_qkv_quantize_pack_766995e25961b5cee390_binding.cu",
        ],
        "template": "minimax_h3_qkv_pack_mxfp8",
        "ffi_entry": "run",
        "compile_flags": ["--use_fast_math"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "out_q"],
            ["buffer", "out_sf"],
            ["parameter", "M"],
            ["parameter", "token_stride"],
            ["parameter", "head_stride"],
            ["parameter", "ROWS_PER_DESTINATION"],
            ["parameter", "SCALE_STRIDE"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch_grid_rule": {
            "kind": "pack_warps_2d_plus_padding",
            "tokens_per_warp": 16,
            "warps_per_cta": 8,
        },
        "launch_block": [256, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
        "use_pdl": False,
        "closure": {
            "sm100a": "a1489ee1821c4d09ae1a4942455209ab8fd5211005e31672b0a6790d836a99b6",
            "sm103a": "bae384f17cfba35e5a5368818498d3ea68d10809fdd09c907bc1f18bd1f9e822",
        },
    },
    "cake_minimax_h3_qkv_quantize_pack_a2de736b0ea8278e8c65": {
        "sources": [
            "cake_minimax_h3_qkv_quantize_pack_a2de736b0ea8278e8c65_kernel.cu",
            "cake_minimax_h3_qkv_quantize_pack_a2de736b0ea8278e8c65_binding.cu",
        ],
        "template": "minimax_h3_qkv_pack_nvfp4",
        "ffi_entry": "run",
        "compile_flags": ["--use_fast_math"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "out_global_scale"],
            ["buffer", "out_q"],
            ["buffer", "out_sf"],
            ["parameter", "M"],
            ["parameter", "token_stride"],
            ["parameter", "head_stride"],
            ["parameter", "ROWS_PER_DESTINATION"],
            ["parameter", "SCALE_STRIDE"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch_grid_rule": {
            "kind": "pack_warps_2d_plus_padding",
            "tokens_per_warp": 16,
            "warps_per_cta": 8,
        },
        "launch_block": [256, 1, 1],
        "cluster": [1, 1, 1],
        "dynamic_smem_bytes": 0,
        "use_pdl": False,
        "closure": {
            "sm100a": "b2a654a5569d8604b30eaa3269f520d6b1c0c185072b730aac39328e2ec8b54d",
            "sm103a": "bf5af4f9d69d3b0a67283b88066d1baf23a1b2e0215905a223a643278461886e",
        },
    },
}
# "<format>:<P>" -> {"module": program name, "defines": {"P": P, "HEADS_PER_DESTINATION": 56 // P}}.
ROUTES: dict[str, dict[str, Any]] = {
    "mxfp8:1": {
        "module": "cake_minimax_h3_qkv_quantize_pack_766995e25961b5cee390",
        "defines": {"P": 1, "HEADS_PER_DESTINATION": 56},
    },
    "mxfp8:2": {
        "module": "cake_minimax_h3_qkv_quantize_pack_766995e25961b5cee390",
        "defines": {"P": 2, "HEADS_PER_DESTINATION": 28},
    },
    "mxfp8:4": {
        "module": "cake_minimax_h3_qkv_quantize_pack_766995e25961b5cee390",
        "defines": {"P": 4, "HEADS_PER_DESTINATION": 14},
    },
    "mxfp8:8": {
        "module": "cake_minimax_h3_qkv_quantize_pack_766995e25961b5cee390",
        "defines": {"P": 8, "HEADS_PER_DESTINATION": 7},
    },
    "nvfp4:1": {
        "module": "cake_minimax_h3_qkv_quantize_pack_a2de736b0ea8278e8c65",
        "defines": {"P": 1, "HEADS_PER_DESTINATION": 56},
    },
    "nvfp4:2": {
        "module": "cake_minimax_h3_qkv_quantize_pack_a2de736b0ea8278e8c65",
        "defines": {"P": 2, "HEADS_PER_DESTINATION": 28},
    },
    "nvfp4:4": {
        "module": "cake_minimax_h3_qkv_quantize_pack_a2de736b0ea8278e8c65",
        "defines": {"P": 4, "HEADS_PER_DESTINATION": 14},
    },
    "nvfp4:8": {
        "module": "cake_minimax_h3_qkv_quantize_pack_a2de736b0ea8278e8c65",
        "defines": {"P": 8, "HEADS_PER_DESTINATION": 7},
    },
}


def minimax_h3_qkv_pack_target(device: torch.device) -> MiniMaxH3QkvPackTarget:
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability == (10, 0):
        return "sm100a"
    if capability == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "MiniMax-H3 QKV quantize-and-pack requires exact compute capability "
        f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
    )


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_minimax_h3_qkv_quantize_pack"
    checkout = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "cake_minimax_h3_qkv_quantize_pack"
    )
    for candidate in (installed, checkout):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "MiniMax-H3 QKV quantize-and-pack CUDA sources were not found. Checked:\n"
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


def _route(P: int, fmt: MiniMaxH3QkvPackFormat) -> dict[str, Any]:
    key = f"{fmt}:{P}"
    try:
        return ROUTES[key]
    except KeyError as exc:
        raise RuntimeError(
            f"no exact MiniMax-H3 QKV quantize-and-pack route for {key}"
        ) from exc


def minimax_h3_qkv_pack_route_record(
    device: torch.device, P: int, fmt: MiniMaxH3QkvPackFormat
) -> dict[str, Any]:
    """``{"P", "format", "target", "module"}`` for one destination partition count and format.

    ``module`` is the generated program record plus its ``name`` and the
    compile-line ``defines`` of this route; the prepared operation binds the
    launch from its ``arg_plan`` and ``launch_grid_rule`` once at preparation.
    """
    target = minimax_h3_qkv_pack_target(device)
    route = _route(P, fmt)
    name = str(route["module"])
    return {
        "P": P,
        "format": fmt,
        "target": target,
        "module": {**MODULES[name], "name": name, "defines": dict(route["defines"])},
    }


@functools.cache
def gen_minimax_h3_qkv_pack_module(
    target: MiniMaxH3QkvPackTarget, P: int, fmt: MiniMaxH3QkvPackFormat
) -> JitSpec:
    """JIT spec of one (target, P, format) instantiation of the format's program."""
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported MiniMax-H3 QKV pack target: {target}")
    route = _route(P, fmt)
    name = str(route["module"])
    record = MODULES[name]
    csrc = _get_csrc_dir()
    defines = [f"-D{key}={value}" for key, value in route["defines"].items()]
    spec = gen_jit_spec(
        name=f"cake_minimax_h3_qkv_quantize_pack_{target}_{fmt}_p{P}_{record['closure'][target][:20]}",
        sources=[csrc / source for source in record["sources"]],
        extra_cuda_cflags=[*_TARGET_FLAGS[target], *record["compile_flags"], *defines],
        extra_include_paths=[csrc, csrc.parent, _get_include_dir()],
        needs_device_linking=True,
    )
    logger.info(
        "Generated MiniMax-H3 QKV quantize-and-pack %s %s P=%d JIT spec: %s",
        target,
        fmt,
        P,
        spec.name,
    )
    return spec


@functools.cache
def load_minimax_h3_qkv_pack_build(
    target: MiniMaxH3QkvPackTarget, P: int, fmt: MiniMaxH3QkvPackFormat
):
    return gen_minimax_h3_qkv_pack_module(target, P, fmt).build_and_load()


def load_minimax_h3_qkv_pack_module(
    device: torch.device, P: int, fmt: MiniMaxH3QkvPackFormat
):
    return load_minimax_h3_qkv_pack_build(minimax_h3_qkv_pack_target(device), P, fmt)


__all__ = [
    "MODULES",
    "ROUTES",
    "MiniMaxH3QkvPackFormat",
    "MiniMaxH3QkvPackTarget",
    "gen_minimax_h3_qkv_pack_module",
    "load_minimax_h3_qkv_pack_build",
    "load_minimax_h3_qkv_pack_module",
    "minimax_h3_qkv_pack_route_record",
    "minimax_h3_qkv_pack_target",
]
