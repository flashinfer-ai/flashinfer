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

# JIT loader for the generated Cake MXFP8 MegaMoE EP16 route.
#
# The generated sources are one architecture-neutral closure (tcgen05/TMEM
# code, no __CUDA_ARCH__ dependence) compiled as one exact-architecture module
# per admitted target. Targets follow FlashInfer's compilation context
# (FLASHINFER_CUDA_ARCH_LIST or the visible devices), restricted to the
# capabilities the kernels are built for.

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from ...compilation_context import CompilationContext
from ...jit import env as jit_env
from ...jit.core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

_OPERATOR_DIR = "cake_mxfp8_megamoe_ep16"
_MANIFEST = "cake_mxfp8_megamoe_ep16_manifest.json"
_SCHEMA = "flashinfer.experimental.source_closure.v2"
_ARCH_FLAGS: dict[tuple[int, int], tuple[str, list[str]]] = {
    (10, 0): ("sm_100a", sm100a_nvcc_flags),
    (10, 3): ("sm_103a", sm103a_nvcc_flags),
}
SUPPORTED_CAPABILITIES = tuple(sorted(_ARCH_FLAGS))


def _get_csrc_root() -> Path:
    package_sources = Path(__file__).resolve().parent / "csrc"
    if (package_sources / _OPERATOR_DIR / _MANIFEST).is_file():
        return package_sources

    raise FileNotFoundError(
        "Cake MXFP8 MegaMoE sources were not found in the installed package or source checkout"
    )


def _get_flashinfer_header_dirs() -> list[Path]:
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer" / "layout.cuh"
    ).is_file():
        return installed

    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer" / "layout.cuh"
    ).is_file():
        return source

    raise FileNotFoundError("FlashInfer JIT headers were not found")


@functools.cache
def _read_manifest() -> tuple[Path, dict[str, Any]]:
    csrc_root = _get_csrc_root()
    manifest = json.loads((csrc_root / _OPERATOR_DIR / _MANIFEST).read_text())
    if manifest.get("schema") != _SCHEMA:
        raise RuntimeError("Cake MXFP8 MegaMoE manifest has an unexpected schema")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or len(sequences) != 1:
        raise RuntimeError("Cake MXFP8 MegaMoE manifest must contain one sequence")
    sequence = sequences[0]
    expected_archs = [
        arch for _capability, (arch, _flags) in sorted(_ARCH_FLAGS.items())
    ]
    if (
        sequence.get("archs") != expected_archs
        or sequence.get("ffi_entry") != "run"
        or sequence.get("setup_ffi_entry") != "setup_tma"
    ):
        raise RuntimeError("Cake MXFP8 MegaMoE manifest has an unexpected ABI")
    units = sequence.get("translation_units")
    if (
        not isinstance(units, dict)
        or not isinstance(units.get("devices"), list)
        or not isinstance(units.get("headers"), list)
        or not isinstance(units.get("binding"), str)
    ):
        raise RuntimeError("Cake MXFP8 MegaMoE manifest has invalid translation units")
    for relative in (*units["devices"], *units["headers"], units["binding"]):
        if not (csrc_root / Path(*Path(relative).parts[1:])).is_file():
            raise FileNotFoundError(f"generated source not found: {relative}")
    return csrc_root, manifest


def supported_capabilities() -> tuple[tuple[int, int], ...]:
    """Admitted capabilities that are also FlashInfer build targets."""

    targets = CompilationContext().TARGET_CUDA_ARCHS
    return tuple(
        capability
        for capability in SUPPORTED_CAPABILITIES
        if (capability[0], f"{capability[1]}a") in targets
    )


def require_supported_capability(capability: tuple[int, int]) -> tuple[int, int]:
    """Return ``capability`` when the route is built for it, else raise naming the fix."""

    capability = (int(capability[0]), int(capability[1]))
    if capability not in _ARCH_FLAGS:
        supported = " or ".join(
            f"{major}.{minor}" for major, minor in SUPPORTED_CAPABILITIES
        )
        raise RuntimeError(
            f"Cake MXFP8 MegaMoE EP16 requires compute capability {supported}, got {capability[0]}.{capability[1]}"
        )
    if capability not in supported_capabilities():
        raise RuntimeError(
            f"Cake MXFP8 MegaMoE EP16 is not a build target for compute capability "
            f"{capability[0]}.{capability[1]}; include {capability[0]}.{capability[1]}a in FLASHINFER_CUDA_ARCH_LIST"
        )
    return capability


def device_arch(capability: tuple[int, int]) -> str:
    """The exact CUDA architecture name (``sm_100a`` / ``sm_103a``) of an admitted capability."""

    return _ARCH_FLAGS[require_supported_capability(capability)][0]


@functools.cache
def gen_cake_mxfp8_megamoe_ep16_module(capability: tuple[int, int]) -> JitSpec:
    """Create the exact-architecture JIT specification for one admitted capability."""

    arch, flags = _ARCH_FLAGS[require_supported_capability(capability)]
    csrc_root, manifest = _read_manifest()
    units = manifest["sequences"][0]["translation_units"]
    sources = [
        csrc_root / Path(*Path(relative).parts[1:])
        for relative in (*units["devices"], units["binding"])
    ]
    operator_dir = csrc_root / _OPERATOR_DIR
    spec = gen_jit_spec(
        name=f"{_OPERATOR_DIR}_{arch}",
        sources=sources,
        extra_cuda_cflags=[*flags, "--use_fast_math"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[csrc_root, operator_dir, *_get_flashinfer_header_dirs()],
    )
    logger.info(f"Generated Cake MXFP8 MegaMoE JIT spec: {spec.name}")
    return spec


@functools.cache
def _build_and_load(capability: tuple[int, int]) -> Any:
    module = gen_cake_mxfp8_megamoe_ep16_module(capability).build_and_load()
    logger.info(f"Loaded Cake MXFP8 MegaMoE EP16 module for capability {capability}")
    return module


def load_cake_mxfp8_megamoe_ep16_module(*, device: Any = None) -> Any:
    """Build or load the generated module for the device's exact architecture."""

    import torch

    from ...utils import get_compute_capability

    resolved = torch.device("cuda") if device is None else torch.device(device)
    return _build_and_load(
        require_supported_capability(get_compute_capability(resolved))
    )


def cute_dsl_source(role: str) -> Path:
    """Path of a generated CuTe DSL device source (``fused`` or ``topk_reduce``)."""

    csrc_root, manifest = _read_manifest()
    relative = manifest["sequences"][0]["cute_dsl"][role]
    return csrc_root / Path(*Path(relative).parts[1:])


__all__ = [
    "SUPPORTED_CAPABILITIES",
    "cute_dsl_source",
    "device_arch",
    "gen_cake_mxfp8_megamoe_ep16_module",
    "load_cake_mxfp8_megamoe_ep16_module",
    "require_supported_capability",
    "supported_capabilities",
]
