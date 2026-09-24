# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-architecture JIT loader for the MiniMax-H3 QKV quantize-and-pack programs."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags

MiniMaxH3QkvPackFormat = Literal["nvfp4", "mxfp8"]

_TARGET = "sm103a"
_TARGET_FLAGS = sm103a_nvcc_flags
# Populated by the Cake generated-program export; the empty table is
# the source-only placeholder and every route lookup fails until it is filled.
# Keys are "<target>:<P>:<format>" -> {"P", "format", "target", "module"}; the
# token count M is a runtime launch parameter, so one route per (P, format)
# serves every M.
_ROUTES: dict[str, dict[str, Any]] = json.loads(r"""{}""")


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


def _verified_source(record: dict[str, str]) -> Path:
    relative = PurePosixPath(record["path"])
    if (
        relative.is_absolute()
        or relative.parts[:2] != ("csrc", "cake_minimax_h3_qkv_quantize_pack")
        or ".." in relative.parts
    ):
        raise RuntimeError(f"invalid MiniMax-H3 QKV pack source path: {relative}")
    path = _get_csrc_dir().joinpath(*relative.parts[2:])
    if not path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 QKV pack source was not installed: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise RuntimeError(f"MiniMax-H3 QKV pack source identity mismatch: {path}")
    return path


def minimax_h3_qkv_pack_route_record(P: int, fmt: MiniMaxH3QkvPackFormat) -> dict:
    """Route record for one destination partition count P and output format.

    The generated program takes the token count M (and the derived
    ROWS_PER_DESTINATION / SCALE_STRIDE) as runtime parameters, so one
    route per (P, format) serves every M.
    """
    key = f"{_TARGET}:{P}:{fmt}"
    try:
        return _ROUTES[key]
    except KeyError as exc:
        raise RuntimeError(
            f"no exact MiniMax-H3 QKV quantize-and-pack route for {key}"
        ) from exc


@functools.cache
def gen_minimax_h3_qkv_pack_module(P: int, fmt: MiniMaxH3QkvPackFormat) -> JitSpec:
    route = minimax_h3_qkv_pack_route_record(P, fmt)
    record = route["module"]
    sources = [_verified_source(item) for item in record["files"]]
    uri = (
        f"cake_minimax_h3_qkv_quantize_pack_{_TARGET}_{fmt}_p{P}_"
        f"{record['closure_sha256']}"
    )
    spec = gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=[*_TARGET_FLAGS, *record["compile_flags"]],
        extra_include_paths=[
            _get_csrc_dir(),
            _get_csrc_dir().parent,
            _get_include_dir(),
        ],
        needs_device_linking=True,
    )
    logger.info(
        "Generated MiniMax-H3 QKV quantize-and-pack %s P=%d JIT spec: %s",
        fmt,
        P,
        spec.name,
    )
    return spec


@functools.cache
def load_minimax_h3_qkv_pack_module(P: int, fmt: MiniMaxH3QkvPackFormat):
    return gen_minimax_h3_qkv_pack_module(P, fmt).build_and_load()


__all__ = [
    "MiniMaxH3QkvPackFormat",
    "gen_minimax_h3_qkv_pack_module",
    "load_minimax_h3_qkv_pack_module",
    "minimax_h3_qkv_pack_route_record",
]
