# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""JIT loader for generated SM120 Sage block-sparse attention kernels."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm120a_nvcc_flags

_GENERATED_ROOT = "csrc/cake_sage_block_sparse_attention"
_MANIFEST_NAME = "cake_sage_block_sparse_attention_manifest.json"
_SPECIALIZATION_NAMES = (
    "HAS_BLOCK_NUMS",
    "BLOCK_SIZES_MODE",
    "FULL_K64_TILES",
    "UNIFORM_NONEMPTY",
    "CONTIGUOUS_BLOCK_INDICES",
)


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_sage_block_sparse_attention"
    if (installed / _MANIFEST_NAME).is_file():
        return installed
    checkout = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "cake_sage_block_sparse_attention"
    )
    if (checkout / _MANIFEST_NAME).is_file():
        return checkout
    raise FileNotFoundError("Cake Sage block-sparse attention sources were not found")


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


@functools.cache
def _manifest() -> dict[str, Any]:
    path = _get_csrc_dir() / _MANIFEST_NAME
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "cake.library_export.v4"
        or value.get("producer") != "cake"
        or value.get("library") != "flashinfer"
        or value.get("name") != "cake_sage_block_sparse_attention"
    ):
        raise RuntimeError("invalid Cake Sage block-sparse attention manifest")
    modules = value.get("modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("empty Cake Sage block-sparse attention module inventory")
    return value


def _key(values: dict[str, Any]) -> tuple[int, ...]:
    if set(values) != set(_SPECIALIZATION_NAMES):
        raise RuntimeError("invalid Sage block-sparse specialization receipt")
    return tuple(int(values[name]) for name in _SPECIALIZATION_NAMES)


def _record(key: tuple[int, ...]) -> dict[str, Any]:
    matches = [
        item
        for item in _manifest()["modules"]
        if item.get("arch") == "sm_120a"
        and _key(dict(item.get("specializations", {}))) == key
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated Sage module for specialization {key}, got {len(matches)}"
        )
    return matches[0]


def _source_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    prefix = Path(_GENERATED_ROOT)
    try:
        suffix = relative.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(
            f"generated source escaped {_GENERATED_ROOT}: {relative_path}"
        ) from exc
    path = _get_csrc_dir() / suffix
    inventory = {item["path"]: item for item in _manifest()["files"]}
    receipt = inventory.get(relative.as_posix())
    if not isinstance(receipt, dict) or not path.is_file():
        raise FileNotFoundError(
            f"generated source is absent from the manifest: {relative_path}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != receipt.get("sha256"):
        raise RuntimeError(f"generated source hash mismatch: {relative_path}")
    return path


@functools.cache
def gen_cake_sage_block_sparse_attention_module(
    has_block_nums: int,
    block_sizes_mode: int,
    full_k64_tiles: int,
    uniform_nonempty: int,
    contiguous_block_indices: int,
) -> JitSpec:
    key = (
        int(has_block_nums),
        int(block_sizes_mode),
        int(full_k64_tiles),
        int(uniform_nonempty),
        int(contiguous_block_indices),
    )
    record = _record(key)
    units = record["translation_units"]
    spec = gen_jit_spec(
        name=str(record["name"]),
        sources=[_source_path(units["device"]), _source_path(units["binding"])],
        extra_cuda_cflags=[*sm120a_nvcc_flags, *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[_get_csrc_dir().parent, _get_include_dir()],
    )
    logger.info("Generated Cake Sage block-sparse attention JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_sage_block_sparse_attention_module(
    has_block_nums: int,
    block_sizes_mode: int,
    full_k64_tiles: int,
    uniform_nonempty: int,
    contiguous_block_indices: int,
):
    key = (
        int(has_block_nums),
        int(block_sizes_mode),
        int(full_k64_tiles),
        int(uniform_nonempty),
        int(contiguous_block_indices),
    )
    spec = gen_cake_sage_block_sparse_attention_module(*key)
    module = spec.build_and_load()
    return module, _record(key)


def build_all_cake_sage_block_sparse_attention_modules() -> dict[str, str]:
    result = {}
    for record in _manifest()["modules"]:
        key = _key(dict(record["specializations"]))
        spec = gen_cake_sage_block_sparse_attention_module(*key)
        module = spec.build_and_load()
        if not hasattr(module, str(record["ffi_entry"])):
            raise RuntimeError(f"built module {record['name']} lacks its FFI entry")
        path = spec.get_library_path().resolve(strict=True)
        result[str(record["name"])] = str(path)
    return result


__all__ = [
    "build_all_cake_sage_block_sparse_attention_modules",
    "gen_cake_sage_block_sparse_attention_module",
    "load_cake_sage_block_sparse_attention_module",
]
