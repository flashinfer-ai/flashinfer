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
_SOURCE_FILE = f"{_GENERATED_DIR}/cake_sampling_kernels.cu"
_MANIFEST_FILE = f"{_GENERATED_DIR}/manifest.json"
_BINDING_HEADER = "cake_sampling_binding.cuh"
# The kernels use thread-block clusters, distributed shared memory, programmatic dependent launch
# and redux.sync only, i.e. the sm_90 feature set, so one frozen source serves every compute
# capability 9.x / 10.x / 11.x / 12.x device.  It is compiled once into a single fatbin with one
# -gencode per target architecture; the targets come from FlashInfer's ``CompilationContext``:
# ``FLASHINFER_CUDA_ARCH_LIST`` when set (AOT builds on hosts without a GPU), otherwise the
# capabilities of the visible devices.
SUPPORTED_MAJOR_VERSIONS: tuple[int, ...] = (9, 10, 11, 12)
_MANIFEST_KEYS = {
    "buckets",
    "codegen_arch",
    "compile_flags",
    "kernel_count",
    "kernel_symbols",
    "min_compute_capability",
    "schema_version",
    "semantics",
    "slab_entries",
    "source_sha256",
    "stage1",
    "stage23",
    "tma_abi",
}
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


def _reject_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise RuntimeError(f"radix sampling manifest has duplicate key {key!r}")
        document[key] = value
    return document


@functools.cache
def load_manifest() -> dict[str, Any]:
    """Load and verify the frozen manifest (``csrc/cake_sampling/generated/manifest.json``)."""
    csrc = _get_csrc_dir()
    source = csrc / _SOURCE_FILE
    manifest_path = csrc / _MANIFEST_FILE
    binding = csrc / _BINDING_HEADER
    missing = [p.name for p in (source, manifest_path, binding) if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"radix sampling source package is incomplete: missing {', '.join(missing)}"
        )
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("radix sampling manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise RuntimeError("radix sampling manifest schema is invalid")
    if manifest["schema_version"] != 1 or manifest["tma_abi"] != "pointer":
        raise RuntimeError("radix sampling manifest identity is invalid")
    min_cc = manifest["min_compute_capability"]
    if (
        not isinstance(min_cc, list)
        or len(min_cc) != 2
        or not all(isinstance(v, int) for v in min_cc)
        or tuple(min_cc) != (min(SUPPORTED_MAJOR_VERSIONS), 0)
    ):
        raise RuntimeError(
            "radix sampling manifest min_compute_capability does not cover the compiled targets"
        )
    if not re.fullmatch(r"sm_[0-9]+a", str(manifest["codegen_arch"])):
        raise RuntimeError("radix sampling manifest codegen_arch is invalid")
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != manifest["source_sha256"]:
        raise RuntimeError("radix sampling source identity is invalid")
    symbols = list(manifest["kernel_symbols"])
    stage_symbols = [v["symbol"] for v in manifest["stage1"]] + [
        v["symbol"] for v in manifest["stage23"]
    ]
    if symbols != stage_symbols or len(symbols) != manifest["kernel_count"]:
        raise RuntimeError("radix sampling manifest kernel inventory is inconsistent")
    for symbol in symbols:
        definitions = re.findall(
            rb"(?<![A-Za-z0-9_])" + re.escape(symbol.encode()) + rb"\(", source_bytes
        )
        if len(definitions) != 1:
            raise RuntimeError(
                f"radix sampling source does not define {symbol} exactly once"
            )
    return manifest


def _binding_source(manifest: dict[str, Any]) -> str:
    min_major, min_minor = manifest["min_compute_capability"]
    stage1 = " ".join(
        f"X({v['symbol']}, {v['cluster']}, {v['ept']}, {1 if v['stream'] else 0}, "
        f"{v['block_threads']}, {v['dynamic_smem_bytes']})"
        for v in manifest["stage1"]
    )
    stage23 = " ".join(
        f"X({v['symbol']}, {v['threads']}, {v['items']}, {v['dynamic_smem_bytes']})"
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
#define CAKE_SAMPLING_STAGE1_TABLE(X) {stage1}
#define CAKE_SAMPLING_STAGE23_TABLE(X) {stage23}
#include "{_BINDING_HEADER}"
"""


def _module_identity(manifest: dict[str, Any]) -> str:
    """JIT module name: sealed over the frozen source, manifest and binding.

    The target architectures are not part of the name: FlashInfer's JIT workspace directory is
    already keyed by the ``CompilationContext`` target set, so one name maps to one fatbin per
    target set.
    """
    csrc = _get_csrc_dir()
    digest = hashlib.sha256()
    digest.update((csrc / _SOURCE_FILE).read_bytes())
    digest.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    digest.update((csrc / _BINDING_HEADER).read_bytes())
    digest.update(_binding_source(manifest).encode())
    return f"cake_sampling_{digest.hexdigest()[:20]}"


def get_cake_sampling_uri() -> str:
    return _module_identity(load_manifest())


@functools.cache
def gen_cake_sampling_module() -> JitSpec:
    """One JIT spec compiling the frozen source for every ``CompilationContext`` target."""
    manifest = load_manifest()
    csrc = _get_csrc_dir()
    uri = _module_identity(manifest)
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
    "supported_capabilities",
    "supported_capability",
    "target_capabilities",
]
