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
import hashlib
import json
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
_MANIFEST_FILE = "manifest.json"
_BINDING_HEADER = "cake_megamoe_topk_reduce_binding.cuh"
_KERNEL_SYMBOL = "kernel_cake_megamoe_workspace_topk_reduce_bfloat16_h4096_k6"
# One frozen Cake export per datacenter-Blackwell target.  The exported
# translation units are arch-neutral, but each one is compiled with its own
# ``-gencode`` and its module only runs on the exact compute capability it was
# published for; ``arch`` keys match the Cake exporter profiles.
_ARCHS: dict[str, dict[str, Any]] = {
    "sm_100a": {
        "dir": "sm100a",
        "capability": (10, 0),
        "nvcc_flags": sm100a_nvcc_flags,
        "source_sha256": "3b83507e0d50dd5089389650c692fd4c17283ebe1f371a62ccb2a7d07b854dda",
    },
    "sm_103a": {
        "dir": "sm103a",
        "capability": (10, 3),
        "nvcc_flags": sm103a_nvcc_flags,
        "source_sha256": "3b83507e0d50dd5089389650c692fd4c17283ebe1f371a62ccb2a7d07b854dda",
    },
}
_CAPABILITY_TO_ARCH = {
    record["capability"]: arch for arch, record in _ARCHS.items()
}
_SUPPORTED_CAPABILITIES = tuple(sorted(_CAPABILITY_TO_ARCH))
_MANIFEST_KEYS = {
    "arch",
    "compile_flags",
    "constraints",
    "kernel_count",
    "kernel_symbols",
    "launch",
    "schema_version",
    "source_sha256",
    "tma_abi",
}
_CONSTRAINTS = {
    "capacities": [256, 4096],
    "dtype": "bfloat16",
    "hidden_size": 4096,
    "top_k": 6,
}
_LAUNCH = {
    "block_threads": 256,
    "dynamic_smem_bytes": 0,
    "grid_x": "4 * num_tokens",
}
_LOADED_MODULES: dict[str, Any] = {}


def supported_capabilities() -> tuple[tuple[int, int], ...]:
    """Exact compute capabilities with a frozen reducer export."""

    return _SUPPORTED_CAPABILITIES


def resolve_arch(device: torch.device | int | None = None) -> str:
    """Return the frozen export arch (``sm_100a`` / ``sm_103a``) for ``device``.

    Raises ``NotImplementedError`` when the device has no frozen export; the
    caller is expected to fall back to the CuTeDSL terminal reducer.
    """

    if device is None:
        device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    try:
        return _CAPABILITY_TO_ARCH[capability]
    except KeyError:
        raise NotImplementedError(
            "the frozen MegaMoE TopK reducer is published for compute "
            f"capabilities {list(_SUPPORTED_CAPABILITIES)}, got {capability}"
        ) from None


def _check_arch(arch: str) -> dict[str, Any]:
    try:
        return _ARCHS[arch]
    except KeyError:
        raise ValueError(
            f"unknown frozen MegaMoE TopK reducer arch {arch!r}; "
            f"known: {sorted(_ARCHS)}"
        ) from None


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_megamoe_topk_reduce"
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_megamoe_topk_reduce"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "frozen MegaMoE TopK-reduce sources were not found. Checked:\n"
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


def _reject_duplicate_manifest_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise RuntimeError(
                f"MegaMoE TopK-reduce manifest has duplicate key {key!r}"
            )
        document[key] = value
    return document


def _program_source(arch: str) -> tuple[Path, dict[str, Any]]:
    record = _check_arch(arch)
    csrc_dir = _get_csrc_dir()
    arch_dir = csrc_dir / record["dir"]
    source = arch_dir / _SOURCE_FILE
    manifest_path = arch_dir / _MANIFEST_FILE
    binding_header = csrc_dir / _BINDING_HEADER
    missing = [
        str(path.relative_to(csrc_dir))
        for path in (source, manifest_path, binding_header)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"MegaMoE TopK-reduce {arch} source package is incomplete: missing "
            + ", ".join(missing)
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_manifest_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"MegaMoE TopK-reduce {arch} manifest is invalid JSON"
        ) from error

    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != record["source_sha256"]:
        raise RuntimeError(f"MegaMoE TopK-reduce {arch} source identity is invalid")
    expected = {
        "schema_version": 1,
        "arch": arch,
        "compile_flags": [],
        "tma_abi": "pointer",
        "kernel_count": 1,
        "launch": _LAUNCH,
        "constraints": _CONSTRAINTS,
        "kernel_symbols": [_KERNEL_SYMBOL],
        "source_sha256": record["source_sha256"],
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _MANIFEST_KEYS
        or manifest != expected
    ):
        raise RuntimeError(f"MegaMoE TopK-reduce {arch} manifest identity is invalid")
    if source_bytes.count(_KERNEL_SYMBOL.encode()) != 1:
        raise RuntimeError(
            f"MegaMoE TopK-reduce {arch} source does not define exactly one "
            "frozen kernel symbol"
        )
    return source, manifest


def _module_identity(
    arch: str, source: Path, manifest: dict[str, Any], binding_header: Path
) -> str:
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    digest.update(binding_header.read_bytes())
    digest.update(_binding_source(arch).encode())
    digest.update(arch.encode())
    return f"cake_megamoe_topk_reduce_{arch.replace('_', '')}_{digest.hexdigest()[:20]}"


def _binding_source(arch: str) -> str:
    record = _check_arch(arch)
    major, minor = record["capability"]
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#define CAKE_MEGAMOE_TOPK_REDUCE_BODY_FILE "{record["dir"]}/{_SOURCE_FILE}"
#define CAKE_MEGAMOE_TOPK_REDUCE_KERNEL {_KERNEL_SYMBOL}
#define CAKE_MEGAMOE_TOPK_REDUCE_THREADS {_LAUNCH["block_threads"]}
#define CAKE_MEGAMOE_TOPK_REDUCE_SMEM_BYTES {_LAUNCH["dynamic_smem_bytes"]}
#define CAKE_MEGAMOE_TOPK_REDUCE_CC_MAJOR {major}
#define CAKE_MEGAMOE_TOPK_REDUCE_CC_MINOR {minor}

#include "{_BINDING_HEADER}"
"""


def get_cake_megamoe_topk_reduce_uri(arch: str | None = None) -> str:
    if arch is None:
        arch = resolve_arch()
    source, manifest = _program_source(arch)
    return _module_identity(arch, source, manifest, _get_csrc_dir() / _BINDING_HEADER)


@functools.cache
def gen_cake_megamoe_topk_reduce_module(arch: str = "sm_100a") -> JitSpec:
    record = _check_arch(arch)
    source, manifest = _program_source(arch)
    csrc_dir = _get_csrc_dir()
    uri = _module_identity(arch, source, manifest, csrc_dir / _BINDING_HEADER)
    binding = (
        jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_megamoe_topk_reduce_binding.cu"
    )
    write_if_different(binding, _binding_source(arch))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=record["nvcc_flags"],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
        use_fast_math=False,
    )
    logger.info("Generated frozen MegaMoE TopK-reduce JIT spec: %s", spec.name)
    return spec


def load_cake_megamoe_topk_reduce_module(arch: str | None = None):
    if arch is None:
        arch = resolve_arch()
    module = _LOADED_MODULES.get(arch)
    if module is None:
        module = gen_cake_megamoe_topk_reduce_module(arch).build_and_load()
        _LOADED_MODULES[arch] = module
        logger.info("Loaded frozen MegaMoE TopK-reduce module (%s)", arch)
    return module


def is_cake_megamoe_topk_reduce_module_loaded(
    device: torch.device | int | None = None,
) -> bool:
    """Whether the reducer for ``device`` can launch without lazy build/load work."""

    try:
        arch = resolve_arch(device)
    except NotImplementedError:
        return False
    return arch in _LOADED_MODULES


def get_cake_megamoe_topk_reduce_module(device: torch.device | int | None = None):
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
]
