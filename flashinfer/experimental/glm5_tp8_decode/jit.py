# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Source-addressed, JIT-only CUDA modules for the B300 implementation."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec, sm103a_nvcc_flags

_CSRC = Path(__file__).resolve().parent / "csrc"


def _header_dirs() -> list[Path]:
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if not (source[0] / "tvm_ffi_utils.h").is_file():
        raise FileNotFoundError("FlashInfer binding headers were not found")
    return source


def _digest(family: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"sm103a;precise-math;relaxed-constexpr;abi1")
    paths = sorted((_CSRC / family).rglob("*")) + sorted(_CSRC.glob("*.h"))
    paths += [_CSRC / f"{family}_binding.cu"]
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(_CSRC)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


@functools.cache
def module_spec(family: str) -> JitSpec:
    if family not in ("moe", "attention"):
        raise ValueError(f"unsupported kernel family: {family}")
    kernels = sorted((_CSRC / family).glob("*.cu"))
    binding = _CSRC / f"{family}_binding.cu"
    if not kernels or not binding.is_file():
        raise FileNotFoundError(
            f"GLM-5 TP8 {family} CUDA sources are absent from this FlashInfer "
            "installation (release wheels omit experimental CUDA sources). Install a "
            "nightly wheel from https://flashinfer.ai/whl or install FlashInfer from "
            "source."
        )
    return gen_jit_spec(
        name=f"glm5_tp8_{family}_sm103a_{_digest(family)}",
        sources=[*kernels, binding],
        extra_include_paths=[_CSRC, *_header_dirs()],
        extra_cuda_cflags=[
            *sm103a_nvcc_flags,
            "--expt-relaxed-constexpr",
            "-ftz=false",
            "-prec-div=true",
            "-prec-sqrt=true",
        ],
        use_fast_math=False,
    )


@functools.cache
def load_module(family: str):
    return module_spec(family).build_and_load()
