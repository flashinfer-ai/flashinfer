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
from typing import Any

import torch

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)
from .utils import write_if_different

_SOURCE_FILE = "cake_megamoe_topk_reduce_kernels.cu"
_BINDING_HEADER = "cake_megamoe_topk_reduce_binding.cuh"
_KERNEL_SYMBOL = "kernel_cake_megamoe_workspace_topk_reduce_bfloat16_h4096_k6"
# One arch-neutral generated source serves every datacenter-Blackwell target;
# each module compiles it with its own ``-gencode`` and runs on the exact
# compute capability it was built for.
_ARCHS: dict[str, dict[str, Any]] = {
    "sm_100a": {"capability": (10, 0), "nvcc_flags": sm100a_nvcc_flags},
    "sm_103a": {"capability": (10, 3), "nvcc_flags": sm103a_nvcc_flags},
}
_CAPABILITY_TO_ARCH = {record["capability"]: arch for arch, record in _ARCHS.items()}
_SUPPORTED_CAPABILITIES = tuple(sorted(_CAPABILITY_TO_ARCH))
_LOADED_MODULES: dict[str, Any] = {}

_BINDING_SOURCE = f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "{_BINDING_HEADER}"
"""


def supported_capabilities() -> tuple[tuple[int, int], ...]:
    """Exact compute capabilities the reducer is built for."""

    return _SUPPORTED_CAPABILITIES


def _device_index(device: torch.device | int | str | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    index = torch.device(device).index
    return torch.cuda.current_device() if index is None else index


@functools.cache
def _device_capability(index: int) -> tuple[int, int]:
    """Compute capability of one device ordinal, queried once per process."""

    return tuple(torch.cuda.get_device_capability(index))


def supports_device(device: torch.device | int | str | None = None) -> bool:
    """Whether ``device`` has an exact reducer build (no device query per call)."""

    return _device_capability(_device_index(device)) in _CAPABILITY_TO_ARCH


def resolve_arch(device: torch.device | int | str | None = None) -> str:
    """Return the reducer arch (``sm_100a`` / ``sm_103a``) for ``device``.

    Raises ``NotImplementedError`` when the device has no reducer build; the
    caller is expected to fall back to the CuTeDSL terminal reducer.
    """

    capability = _device_capability(_device_index(device))
    try:
        return _CAPABILITY_TO_ARCH[capability]
    except KeyError:
        raise NotImplementedError(
            "the MegaMoE TopK reducer is published for compute "
            f"capabilities {list(_SUPPORTED_CAPABILITIES)}, got {capability}"
        ) from None


def _check_arch(arch: str) -> dict[str, Any]:
    try:
        return _ARCHS[arch]
    except KeyError:
        raise ValueError(
            f"unknown MegaMoE TopK reducer arch {arch!r}; known: {sorted(_ARCHS)}"
        ) from None


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_megamoe_topk_reduce"
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_megamoe_topk_reduce"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "MegaMoE TopK-reduce sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def get_cake_megamoe_topk_reduce_uri(arch: str | None = None) -> str:
    if arch is None:
        arch = resolve_arch()
    _check_arch(arch)
    return f"cake_megamoe_topk_reduce_{arch.replace('_', '')}"


@functools.cache
def gen_cake_megamoe_topk_reduce_module(arch: str = "sm_100a") -> JitSpec:
    record = _check_arch(arch)
    csrc_dir = _get_csrc_dir()
    binding = (
        jit_env.FLASHINFER_GEN_SRC_DIR
        / "cake_megamoe_topk_reduce"
        / "cake_megamoe_topk_reduce_binding.cu"
    )
    write_if_different(binding, _BINDING_SOURCE)
    spec = gen_jit_spec(
        name=get_cake_megamoe_topk_reduce_uri(arch),
        sources=[binding],
        extra_cuda_cflags=record["nvcc_flags"],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
        use_fast_math=False,
    )
    logger.info("Generated MegaMoE TopK-reduce JIT spec: %s", spec.name)
    return spec


def load_cake_megamoe_topk_reduce_module(arch: str | None = None):
    if arch is None:
        arch = resolve_arch()
    module = _LOADED_MODULES.get(arch)
    if module is None:
        module = gen_cake_megamoe_topk_reduce_module(arch).build_and_load()
        _LOADED_MODULES[arch] = module
        logger.info("Loaded MegaMoE TopK-reduce module (%s)", arch)
    return module


def is_cake_megamoe_topk_reduce_module_loaded(
    device: torch.device | int | str | None = None,
) -> bool:
    """Whether the reducer for ``device`` can launch without lazy build/load work."""

    try:
        arch = resolve_arch(device)
    except NotImplementedError:
        return False
    return arch in _LOADED_MODULES


def get_cake_megamoe_topk_reduce_module(device: torch.device | int | str | None = None):
    return load_cake_megamoe_topk_reduce_module(resolve_arch(device))


def run_cake_megamoe_topk_reduce(
    partials: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
) -> None:
    """Launch the reducer on ``partials``' current CUDA stream."""

    stream = torch.cuda.current_stream(device=partials.device).cuda_stream
    get_cake_megamoe_topk_reduce_module(partials.device).run(
        partials,
        out,
        num_tokens,
        stream,
    )


__all__ = [
    "gen_cake_megamoe_topk_reduce_module",
    "get_cake_megamoe_topk_reduce_module",
    "get_cake_megamoe_topk_reduce_uri",
    "load_cake_megamoe_topk_reduce_module",
    "is_cake_megamoe_topk_reduce_module_loaded",
    "resolve_arch",
    "run_cake_megamoe_topk_reduce",
    "supported_capabilities",
    "supports_device",
]
