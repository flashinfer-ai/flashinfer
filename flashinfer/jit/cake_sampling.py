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

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from ..compilation_context import CompilationContext
from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger
from .utils import write_if_different

_GENERATED_DIR = "generated"
# The root translation unit; its kernel bodies live in the ``source_files`` parts the manifest
# lists (``cake_sampling_kernels_part<N>.cuh``, each under the repository's 5 MiB file limit),
# which the root includes in order.
_SOURCE_FILE = f"{_GENERATED_DIR}/cake_sampling_kernels.cu"
_MANIFEST_FILE = f"{_GENERATED_DIR}/manifest.json"
_BINDING_HEADER = "cake_sampling_binding.cuh"
_MODULE_NAME = "cake_sampling"
# The kernels use thread-block clusters, distributed shared memory, programmatic dependent launch
# and redux.sync only, i.e. the sm_90 feature set, so one frozen source serves every compute
# capability 9.x / 10.x / 11.x / 12.x device.  It is compiled once into a single fatbin with one
# -gencode per target architecture; the targets come from FlashInfer's ``CompilationContext``:
# ``FLASHINFER_CUDA_ARCH_LIST`` when set (AOT builds on hosts without a GPU), otherwise the
# capabilities of the visible devices.  A stage-2/3 static form is compiled only for the targets
# whose host dispatches it (the frozen source guards the other forms out per ``__CUDA_ARCH__``).
SUPPORTED_MAJOR_VERSIONS: tuple[int, ...] = (9, 10, 11, 12)
_LOADED: dict[str, Any] = {}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_sampling"
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_sampling"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "frozen radix sampling sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _capability_of(major: Any, minor: Any) -> tuple[int, int]:
    """``(major, minor)`` of a ``CompilationContext.TARGET_CUDA_ARCHS`` entry (``(9, "0a")`` -> ``(9, 0)``)."""
    return int(major), int(re.sub(r"[a-z]+$", "", str(minor)))


@functools.cache
def target_capabilities() -> tuple[tuple[int, int], ...]:
    """Compute capabilities the module is built for, in ascending order.

    Read once per process from ``CompilationContext`` (``FLASHINFER_CUDA_ARCH_LIST`` when set,
    else the visible devices) and restricted to :data:`SUPPORTED_MAJOR_VERSIONS`.  Empty when the
    process has neither an arch list nor a CUDA device.
    """
    context = CompilationContext()
    return tuple(
        sorted(
            {
                _capability_of(major, minor)
                for major, minor in context.TARGET_CUDA_ARCHS
                if int(major) in SUPPORTED_MAJOR_VERSIONS
            }
        )
    )


def supported_capability(capability: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` when the frozen kernels are built for it, else ``None``.

    A device whose capability is not among the build targets (a major outside
    :data:`SUPPORTED_MAJOR_VERSIONS`, or an architecture left out of ``FLASHINFER_CUDA_ARCH_LIST``)
    takes the ``top_k_first`` fallback instead of failing at launch.
    """
    key = (int(capability[0]), int(capability[1]))
    return key if key in target_capabilities() else None


def supported_capabilities() -> tuple[tuple[int, int], ...]:
    """Capabilities the frozen kernels are built for in this process."""
    return target_capabilities()


def nvcc_flags() -> list[str]:
    """``-gencode`` flags for every build target (plus FlashInfer's common flags).

    Raises ``RuntimeError`` when no target has a supported major version.
    """
    return CompilationContext().get_nvcc_flags_list(
        supported_major_versions=list(SUPPORTED_MAJOR_VERSIONS)
    )


@functools.cache
def load_manifest() -> dict[str, Any]:
    """The frozen manifest (``csrc/cake_sampling/generated/manifest.json``).

    It carries the launch resources of every frozen kernel (``stage1`` / ``stage23`` rows: symbol,
    threads, dynamic shared memory, cluster size, build flags, the capabilities a stage-2/3 form is
    dispatched on), the slab geometry and the names of the frozen source files.  The generator
    writes it together with the source.
    """
    return json.loads((_get_csrc_dir() / _MANIFEST_FILE).read_text(encoding="utf-8"))


@functools.cache
def stage23_flags_by_capability() -> dict[Optional[tuple[int, int]], int]:
    """``variant_flags`` of the stage-2/3 form each compute capability dispatches (manifest
    ``capabilities``); the ``None`` key is the form every capability not listed runs."""
    table: dict[Optional[tuple[int, int]], int] = {}
    for v in load_manifest()["stage23"]:
        flags = int(v["variant_flags"])
        if v["capabilities"] is None:
            table[None] = flags
        else:
            for major, minor in v["capabilities"]:
                table[(int(major), int(minor))] = flags
    return table


def _binding_source(manifest: dict[str, Any]) -> str:
    min_major, min_minor = manifest["min_compute_capability"]
    stage1 = " ".join(
        f"X({v['symbol']}, {v['cluster']}, {v['ept']}, {1 if v['stream'] else 0}, "
        f"{v['block_threads']}, {v['dynamic_smem_bytes']}, {1 if v['fused_tail'] else 0}, "
        f"{1 if v['fused_block_tail'] else 0}, {1 if v['coarse_sample'] else 0}, "
        f"{1 if v['spec_sample'] else 0}, {1 if v['slab_tail'] else 0}, "
        f"{1 if v['coarse_push'] else 0})"
        for v in manifest["stage1"]
    )
    stage23 = " ".join(
        f"X({v['symbol']}, {v['threads']}, {v['items']}, {int(v['variant_flags'])}, "
        f"{v['dynamic_smem_bytes']})"
        for v in manifest["stage23"]
    )
    return f"""\
/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */
#define CAKE_SAMPLING_BODY_FILE "{_SOURCE_FILE}"
#define CAKE_SAMPLING_MIN_MAJOR {min_major}
#define CAKE_SAMPLING_MIN_MINOR {min_minor}
#define CAKE_SAMPLING_SLAB {manifest["slab_entries"]}
#define CAKE_SAMPLING_FUSED_TAIL_KCAP {manifest["fused_tail_kcap"]}
#define CAKE_SAMPLING_FUSED_BLOCK_TAIL_KCAP {manifest["fused_block_tail_kcap"]}
#define CAKE_SAMPLING_STAGE1_TABLE(X) {stage1}
#define CAKE_SAMPLING_STAGE23_TABLE(X) {stage23}
#include "{_BINDING_HEADER}"
"""


def _module_identity(manifest: dict[str, Any]) -> str:
    """JIT module name, sealed over the frozen source parts (manifest order), the manifest, the
    binding header and the rendered binding: ``cake_sampling_`` + 20 hex digits.

    FlashInfer resolves installed AOT artifacts by module name before ninja sees the build inputs,
    so a fixed name could load the artifact of another bundle revision; the content-derived name
    cannot.  The target architectures are not part of the name:
    FlashInfer's JIT workspace directory is already keyed by the ``CompilationContext`` target
    set, so one name maps to one fatbin per target set.
    """
    csrc = _get_csrc_dir()
    digest = hashlib.sha256()
    for name in manifest["source_files"]:
        digest.update((csrc / _GENERATED_DIR / name).read_bytes())
    digest.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    digest.update((csrc / _BINDING_HEADER).read_bytes())
    digest.update(_binding_source(manifest).encode())
    return f"{_MODULE_NAME}_{digest.hexdigest()[:20]}"


@functools.cache
def get_cake_sampling_uri() -> str:
    """Content-derived JIT module name (see :func:`_module_identity`), computed once per process."""
    return _module_identity(load_manifest())


@functools.cache
def gen_cake_sampling_module() -> JitSpec:
    """One JIT spec compiling the frozen source for every ``CompilationContext`` target."""
    manifest = load_manifest()
    csrc = _get_csrc_dir()
    uri = get_cake_sampling_uri()
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_sampling_binding.cu"
    write_if_different(binding, _binding_source(manifest))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=nvcc_flags() + list(manifest["compile_flags"]),
        extra_include_paths=[csrc, csrc.parent, _get_include_dir()],
        use_fast_math=False,
    )
    logger.info(
        "Generated frozen radix sampling JIT spec %s for compute capabilities %s",
        spec.name,
        ", ".join(f"{a}.{b}" for a, b in supported_capabilities()),
    )
    return spec


def load_cake_sampling_module():
    module = _LOADED.get("module")
    if module is None:
        module = gen_cake_sampling_module().build_and_load()
        _LOADED["module"] = module
        logger.info("Loaded frozen radix sampling module")
    return module


def is_cake_sampling_module_loaded() -> bool:
    return "module" in _LOADED


__all__ = [
    "SUPPORTED_MAJOR_VERSIONS",
    "gen_cake_sampling_module",
    "get_cake_sampling_uri",
    "is_cake_sampling_module_loaded",
    "load_cake_sampling_module",
    "load_manifest",
    "nvcc_flags",
    "stage23_flags_by_capability",
    "supported_capabilities",
    "supported_capability",
    "target_capabilities",
]
