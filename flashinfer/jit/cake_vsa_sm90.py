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


"""JIT loader for the generated Cake SM90 (Hopper) variable block-sparse attention kernel.

One source-only manifest lives under ``csrc/cake_vsa_sm90``
(``cake_vsa_sm90_manifest.json``, schema ``cake.library_export.v4``): a single
``sm_90a`` module rendered from the Cake Weave kernel ``vsa_sm90_bf16_fwd`` (BF16 HND,
head_dim 128, 64-token blocks) plus its tvm-ffi binding.  The Python planner that
produces the kernel's tile metadata lives in :mod:`flashinfer.cake_vsa_sm90`.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm90a_nvcc_flags

_GENERATED_ROOT = "csrc/cake_vsa_sm90"
_MANIFEST_NAME = "cake_vsa_sm90_manifest.json"
_ARCH = "sm_90a"
_STAGE = "attention"


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_vsa_sm90"
    if (installed / _MANIFEST_NAME).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_vsa_sm90"
    if (checkout / _MANIFEST_NAME).is_file():
        return checkout
    raise FileNotFoundError("Cake SM90 VSA sources were not found")


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
        or value.get("name") != "cake_vsa_sm90"
    ):
        raise RuntimeError("invalid Cake SM90 VSA manifest")
    modules = value.get("modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("empty Cake SM90 VSA module inventory")
    return value


def _record() -> dict[str, Any]:
    matches = [
        item
        for item in _manifest()["modules"]
        if item.get("arch") == _ARCH
        and dict(item.get("route", {})).get("stage") == _STAGE
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated Cake SM90 VSA module for ({_ARCH}, {_STAGE}), got {len(matches)}"
        )
    return matches[0]


def _source_path(relative_path: str) -> Path:
    manifest = _manifest()
    relative = Path(relative_path)
    try:
        suffix = relative.relative_to(Path(_GENERATED_ROOT))
    except ValueError as exc:
        raise RuntimeError(
            f"generated source escaped {_GENERATED_ROOT}: {relative_path}"
        ) from exc
    path = _get_csrc_dir() / suffix
    inventory = {item["path"]: item for item in manifest["files"]}
    receipt = inventory.get(relative.as_posix())
    if not isinstance(receipt, dict) or not path.is_file():
        raise FileNotFoundError(
            f"generated source is absent from the manifest: {relative_path}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != receipt.get("sha256"):
        raise RuntimeError(f"generated source hash mismatch: {relative_path}")
    return path


def cake_vsa_sm90_launch() -> dict[str, Any]:
    """Launch record of the generated module (block, dynamic SMEM, PDL/cooperative flags)."""
    return dict(_record()["launch"])


@functools.cache
def gen_cake_vsa_sm90_module() -> JitSpec:
    record = _record()
    units = record["translation_units"]
    spec = gen_jit_spec(
        name=f"{record['name']}_{_ARCH}",
        sources=[_source_path(units["device"]), _source_path(units["binding"])],
        extra_cuda_cflags=[*sm90a_nvcc_flags, *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[_get_csrc_dir().parent, _get_include_dir()],
    )
    logger.info("Generated Cake SM90 VSA JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_vsa_sm90_module():
    """Build and load the ``sm_90a`` module; return ``(module, manifest record)``."""
    spec = gen_cake_vsa_sm90_module()
    module = spec.build_and_load()
    return module, _record()


def build_all_cake_vsa_sm90_modules() -> dict[str, str]:
    spec = gen_cake_vsa_sm90_module()
    module = spec.build_and_load()
    record = _record()
    if not hasattr(module, str(record["ffi_entry"])):
        raise RuntimeError(f"built module {record['name']} lacks its FFI entry")
    return {
        f"{record['name']}_{_ARCH}": str(spec.get_library_path().resolve(strict=True))
    }


__all__ = [
    "build_all_cake_vsa_sm90_modules",
    "cake_vsa_sm90_launch",
    "gen_cake_vsa_sm90_module",
    "load_cake_vsa_sm90_module",
]
