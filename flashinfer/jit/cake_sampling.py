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
from typing import Any

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags
from .utils import write_if_different

_SOURCE_FILE = "cake_sampling_kernels.cu"
_MANIFEST_FILE = "manifest.json"
_BINDING_HEADER = "cake_sampling_binding.cuh"
_ARCH_DIRS = {(10, 0): "sm100a", (10, 3): "sm103a"}
_ARCH_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}
_MANIFEST_KEYS = {
    "arch",
    "buckets",
    "compile_flags",
    "kernel_count",
    "kernel_symbols",
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


def arch_dir_for_capability(capability: tuple[int, int]) -> str | None:
    """Return the frozen source directory name for a compute capability, or None."""
    return _ARCH_DIRS.get((int(capability[0]), int(capability[1])))


def _reject_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise RuntimeError(f"radix sampling manifest has duplicate key {key!r}")
        document[key] = value
    return document


@functools.cache
def load_manifest(arch: str) -> dict[str, Any]:
    """Load and verify the frozen manifest for ``arch`` (``sm100a`` or ``sm103a``)."""
    if arch not in _ARCH_FLAGS:
        raise ValueError(f"unsupported radix sampling arch {arch!r}")
    arch_dir = _get_csrc_dir() / arch
    source = arch_dir / _SOURCE_FILE
    manifest_path = arch_dir / _MANIFEST_FILE
    binding = _get_csrc_dir() / _BINDING_HEADER
    missing = [p.name for p in (source, manifest_path, binding) if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"radix sampling source package for {arch} is incomplete: missing {', '.join(missing)}"
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
    if manifest["arch"] != f"sm_{arch[2:-1]}a":
        raise RuntimeError(
            f"radix sampling manifest arch {manifest['arch']!r} does not match {arch}"
        )
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != manifest["source_sha256"]:
        raise RuntimeError(f"radix sampling source identity is invalid for {arch}")
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


def _binding_source(arch: str, manifest: dict[str, Any]) -> str:
    major, minor = {"sm100a": (10, 0), "sm103a": (10, 3)}[arch]
    stage1 = " ".join(
        f"X({v['symbol']}, {v['cluster']}, {v['ept']}, {v['block_threads']}, {v['dynamic_smem_bytes']})"
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
#define CAKE_SAMPLING_BODY_FILE "{arch}/{_SOURCE_FILE}"
#define CAKE_SAMPLING_TARGET_MAJOR {major}
#define CAKE_SAMPLING_TARGET_MINOR {minor}
#define CAKE_SAMPLING_SLAB {manifest["slab_entries"]}
#define CAKE_SAMPLING_STAGE1_TABLE(X) {stage1}
#define CAKE_SAMPLING_STAGE23_TABLE(X) {stage23}
#include "{_BINDING_HEADER}"
"""


def _module_identity(arch: str, manifest: dict[str, Any]) -> str:
    csrc = _get_csrc_dir()
    digest = hashlib.sha256()
    digest.update((csrc / arch / _SOURCE_FILE).read_bytes())
    digest.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    digest.update((csrc / _BINDING_HEADER).read_bytes())
    digest.update(_binding_source(arch, manifest).encode())
    digest.update(arch.encode())
    return f"cake_sampling_{arch}_{digest.hexdigest()[:20]}"


def get_cake_sampling_uri(arch: str) -> str:
    return _module_identity(arch, load_manifest(arch))


@functools.cache
def gen_cake_sampling_module(arch: str) -> JitSpec:
    manifest = load_manifest(arch)
    csrc = _get_csrc_dir()
    uri = _module_identity(arch, manifest)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_sampling_binding.cu"
    write_if_different(binding, _binding_source(arch, manifest))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=list(_ARCH_FLAGS[arch]) + list(manifest["compile_flags"]),
        extra_include_paths=[csrc, csrc.parent, _get_include_dir()],
        use_fast_math=False,
    )
    logger.info("Generated frozen radix sampling JIT spec: %s", spec.name)
    return spec


def load_cake_sampling_module(arch: str):
    module = _LOADED.get(arch)
    if module is None:
        module = gen_cake_sampling_module(arch).build_and_load()
        _LOADED[arch] = module
        logger.info("Loaded frozen radix sampling module for %s", arch)
    return module


def is_cake_sampling_module_loaded(arch: str) -> bool:
    return arch in _LOADED


__all__ = [
    "arch_dir_for_capability",
    "gen_cake_sampling_module",
    "get_cake_sampling_uri",
    "is_cake_sampling_module_loaded",
    "load_cake_sampling_module",
    "load_manifest",
]
