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

FlashKDATrainingTarget = Literal["sm100a", "sm103a"]

# First ten hex digits of SHA256 over the normalized generated body, binding,
# and shared binding header, separated by NUL bytes.
_FLASH_KDA_TRAINING_MODULE_IDENT = "ed397caf5a"
_TARGET_FLAGS: dict[FlashKDATrainingTarget, list[str]] = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("frozen FlashKDA training sources were not found")


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def get_flash_kda_training_uri(target: FlashKDATrainingTarget) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported FlashKDA training target: {target}")
    return f"flash_kda_training_{_FLASH_KDA_TRAINING_MODULE_IDENT}_{target}"


@functools.cache
def gen_flash_kda_training_module(target: FlashKDATrainingTarget) -> JitSpec:
    csrc_dir = _get_csrc_dir()
    binding = csrc_dir / "flashkda_training_forward_v483_binding.cu"
    body = csrc_dir / "flashkda_training_forward_v483.cu"
    common = csrc_dir / "flashkda_binding_common.cuh"
    for source in (binding, body, common):
        if not source.exists():
            raise FileNotFoundError(f"FlashKDA training source not found: {source}")
    spec = gen_jit_spec(
        name=get_flash_kda_training_uri(target),
        sources=[binding],
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            f"-DFLASHINFER_FLASH_KDA_TARGET_MINOR={0 if target == 'sm100a' else 3}",
        ],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
    )
    logger.info(f"Generated FlashKDA training {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def load_flash_kda_training_module(target: FlashKDATrainingTarget):
    module = gen_flash_kda_training_module(target).build_and_load()
    logger.info(f"Loaded FlashKDA training {target} module")
    return module


__all__ = [
    "FlashKDATrainingTarget",
    "gen_flash_kda_training_module",
    "get_flash_kda_training_uri",
    "load_flash_kda_training_module",
]
