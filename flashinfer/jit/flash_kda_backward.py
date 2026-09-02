"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from pathlib import Path
from typing import Literal

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

# First ten hex digits of SHA256 over the normalized C32 body, C16 body, C32
# binding, and C16 binding, separated by NUL bytes. This keeps the JIT cache
# tied to executable content without making the identifier self-referential.
_FLASH_KDA_BACKWARD_MODULE_IDENT = "6af2ba3a09"

FlashKDABackwardTarget = Literal["sm100a", "sm103a"]

_TARGET_FLAGS: dict[FlashKDABackwardTarget, list[str]] = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}


def _get_flash_kda_backward_csrc_dir() -> Path:
    """Locate the frozen backward sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "frozen FlashKDA backward sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    """Locate FlashInfer headers in installed and source checkouts."""

    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR

    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def get_flash_kda_backward_uri(target: FlashKDABackwardTarget) -> str:
    """Return the exact-architecture JIT/AOT module key."""

    return f"flash_kda_backward_{_FLASH_KDA_BACKWARD_MODULE_IDENT}_{target}"


@functools.cache
def gen_flash_kda_backward_module(target: FlashKDABackwardTarget) -> JitSpec:
    """Generate the exact-architecture recurrent backward module."""

    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported FlashKDA backward target: {target}")

    csrc_dir = _get_flash_kda_backward_csrc_dir()
    binding = csrc_dir / "flashkda_backward_binding.cu"
    v483_binding = csrc_dir / "flashkda_backward_v483_binding.cu"
    body = csrc_dir / "flashkda_backward.cu"
    v483_body = csrc_dir / "flashkda_backward_v483.cu"
    common = csrc_dir / "flashkda_binding_common.cuh"
    for source in (binding, v483_binding, body, v483_body, common):
        if not source.exists():
            raise FileNotFoundError(f"FlashKDA backward source not found: {source}")

    spec = gen_jit_spec(
        name=get_flash_kda_backward_uri(target),
        sources=[binding, v483_binding],
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            f"-DFLASHINFER_FLASH_KDA_TARGET_MINOR={0 if target == 'sm100a' else 3}",
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            _get_include_dir(),
        ],
    )
    logger.info(f"Generated FlashKDA backward JIT spec: {spec.name}")
    return spec


@functools.cache
def load_flash_kda_backward_module(target: FlashKDABackwardTarget):
    """Build or load the exact-architecture backward module."""

    module = gen_flash_kda_backward_module(target).build_and_load()
    logger.info(f"Loaded FlashKDA backward {target} module")
    return module


def get_flash_kda_backward_module(target: FlashKDABackwardTarget):
    """Return the loaded module used by the recurrent-KDA backward API."""

    return load_flash_kda_backward_module(target)


__all__ = [
    "FlashKDABackwardTarget",
    "gen_flash_kda_backward_module",
    "get_flash_kda_backward_module",
    "get_flash_kda_backward_uri",
    "load_flash_kda_backward_module",
]
