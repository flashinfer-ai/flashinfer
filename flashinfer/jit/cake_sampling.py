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
import re
import json
from pathlib import Path
from typing import Any, Optional

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
    sm110a_nvcc_flags,
)
from .utils import write_if_different

_GENERATED_DIR = "generated"
_SOURCE_FILE = f"{_GENERATED_DIR}/cake_sampling_kernels.cu"
_MANIFEST_FILE = f"{_GENERATED_DIR}/manifest.json"
_BINDING_HEADER = "cake_sampling_binding.cuh"
# One frozen source serves every supported device; it is compiled once per compute capability
# with that capability's own -gencode flags.  The kernels use thread-block clusters, distributed
# shared memory, programmatic dependent launch and redux.sync, i.e. sm_90-class features only.
_CAPABILITY_FLAGS: dict[tuple[int, int], list[str]] = {
    (9, 0): sm90a_nvcc_flags,
    (10, 0): sm100a_nvcc_flags,
    (10, 3): sm103a_nvcc_flags,
    (10, 7): sm107a_nvcc_flags,
    (11, 0): sm110a_nvcc_flags,
}
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
_LOADED: dict[tuple[int, int], Any] = {}


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


def supported_capability(capability: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` when the frozen kernels are compiled for it, else ``None``."""
    key = (int(capability[0]), int(capability[1]))
    return key if key in _CAPABILITY_FLAGS else None


def supported_capabilities() -> tuple[tuple[int, int], ...]:
    return tuple(sorted(_CAPABILITY_FLAGS))


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
        or any(cap < tuple(min_cc) for cap in _CAPABILITY_FLAGS)
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


def _binding_source(capability: tuple[int, int], manifest: dict[str, Any]) -> str:
    major, minor = capability
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
#define CAKE_SAMPLING_TARGET_MAJOR {major}
#define CAKE_SAMPLING_TARGET_MINOR {minor}
#define CAKE_SAMPLING_SLAB {manifest["slab_entries"]}
#define CAKE_SAMPLING_STAGE1_TABLE(X) {stage1}
#define CAKE_SAMPLING_STAGE23_TABLE(X) {stage23}
#include "{_BINDING_HEADER}"
"""


def _module_identity(capability: tuple[int, int], manifest: dict[str, Any]) -> str:
    csrc = _get_csrc_dir()
    digest = hashlib.sha256()
    digest.update((csrc / _SOURCE_FILE).read_bytes())
    digest.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    digest.update((csrc / _BINDING_HEADER).read_bytes())
    digest.update(_binding_source(capability, manifest).encode())
    digest.update(" ".join(_CAPABILITY_FLAGS[capability]).encode())
    return f"cake_sampling_sm{capability[0]}{capability[1]}_{digest.hexdigest()[:20]}"


def _checked_capability(capability: tuple[int, int]) -> tuple[int, int]:
    key = supported_capability(capability)
    if key is None:
        raise ValueError(
            "frozen radix sampling kernels are not compiled for compute capability "
            f"{capability[0]}.{capability[1]}; supported: "
            + ", ".join(f"{a}.{b}" for a, b in supported_capabilities())
        )
    return key


def get_cake_sampling_uri(capability: tuple[int, int]) -> str:
    return _module_identity(_checked_capability(capability), load_manifest())


@functools.cache
def gen_cake_sampling_module(capability: tuple[int, int]) -> JitSpec:
    capability = _checked_capability(capability)
    manifest = load_manifest()
    csrc = _get_csrc_dir()
    uri = _module_identity(capability, manifest)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_sampling_binding.cu"
    write_if_different(binding, _binding_source(capability, manifest))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=list(_CAPABILITY_FLAGS[capability])
        + list(manifest["compile_flags"]),
        extra_include_paths=[csrc, csrc.parent, _get_include_dir()],
        use_fast_math=False,
    )
    logger.info("Generated frozen radix sampling JIT spec: %s", spec.name)
    return spec


def load_cake_sampling_module(capability: tuple[int, int]):
    capability = _checked_capability(capability)
    module = _LOADED.get(capability)
    if module is None:
        module = gen_cake_sampling_module(capability).build_and_load()
        _LOADED[capability] = module
        logger.info(
            "Loaded frozen radix sampling module for sm_%d%d",
            capability[0],
            capability[1],
        )
    return module


def is_cake_sampling_module_loaded(capability: tuple[int, int]) -> bool:
    return supported_capability(capability) in _LOADED


__all__ = [
    "gen_cake_sampling_module",
    "get_cake_sampling_uri",
    "is_cake_sampling_module_loaded",
    "load_cake_sampling_module",
    "load_manifest",
    "supported_capabilities",
    "supported_capability",
]
