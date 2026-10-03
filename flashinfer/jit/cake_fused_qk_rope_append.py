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
"""JIT loader for the generated Cake fused QK RMSNorm + NeoX RoPE + paged KV append (BF16) kernel.

One source-only manifest lives under ``csrc/cake_fused_qk_rope_append``
(``cake_fused_qk_rope_append_manifest.json``, schema ``cake.library_export.v5``):
two head-configuration stages (``hq8_hkv1`` for ``(Hq, Hkv) = (8, 1)`` and
``hq64_hkv8`` for ``(64, 8)``) rendered for ``sm_90a`` (H100/H200), ``sm_100a``
(B200) and ``sm_103a`` (B300), each with its tvm-ffi binding.  The host wrapper
lives in :mod:`flashinfer.cake_fused_qk_rope_append`.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

_PACKAGE = "cake_fused_qk_rope_append"
_GENERATED_ROOT = f"csrc/{_PACKAGE}"
_MANIFEST_NAME = f"{_PACKAGE}_manifest.json"

STAGES = ("hq8_hkv1", "hq64_hkv8")
ARCHES = ("sm_90a", "sm_100a", "sm_103a")
_NVCC_FLAGS = {
    "sm_90a": sm90a_nvcc_flags,
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def stage_for(num_q_heads: int, num_kv_heads: int) -> str:
    if (num_q_heads, num_kv_heads) == (8, 1):
        return "hq8_hkv1"
    if (num_q_heads, num_kv_heads) == (64, 8):
        return "hq64_hkv8"
    raise ValueError(
        f"cake_fused_qk_rope_append supports (num_q_heads, num_kv_heads) in {{(8, 1), (64, 8)}}, "
        f"got ({num_q_heads}, {num_kv_heads})"
    )


def arch_for(compute_capability: tuple[int, int]) -> str:
    major, minor = compute_capability
    if (major, minor) == (9, 0):
        return "sm_90a"
    if (major, minor) == (10, 0):
        return "sm_100a"
    if (major, minor) == (10, 3):
        return "sm_103a"
    raise ValueError(
        f"cake_fused_qk_rope_append has no cubin for compute capability {major}.{minor}; "
        f"supported: sm_90a (9.0), sm_100a (10.0), sm_103a (10.3)"
    )


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / _PACKAGE
    if (installed / _MANIFEST_NAME).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / _PACKAGE
    if (checkout / _MANIFEST_NAME).is_file():
        return checkout
    raise FileNotFoundError("Cake fused QK RoPE append sources were not found")


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
        value.get("schema") != "cake.library_export.v5"
        or value.get("producer") != "cake"
        or value.get("library") != "flashinfer"
        or value.get("name") != _PACKAGE
    ):
        raise RuntimeError("invalid Cake fused QK RoPE append manifest")
    modules = value.get("modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("empty Cake fused QK RoPE append module inventory")
    return value


def _record(stage: str, arch: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHES}")
    matches = [
        item
        for item in _manifest()["modules"]
        if item.get("arch") == arch
        and dict(item.get("route", {})).get("stage") == stage
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated Cake module for ({arch}, {stage}), got {len(matches)}"
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


def cake_fused_qk_rope_append_record(stage: str, arch: str) -> dict[str, Any]:
    """Manifest record of one generated module (arg plan, launch block, specialization)."""
    return dict(_record(stage, arch))


@functools.cache
def gen_cake_fused_qk_rope_append_module(stage: str, arch: str) -> JitSpec:
    record = _record(stage, arch)
    units = record["translation_units"]
    spec = gen_jit_spec(
        name=f"{record['name']}_{arch}",
        sources=[_source_path(units["device"]), _source_path(units["binding"])],
        extra_cuda_cflags=[*_NVCC_FLAGS[arch], *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[_get_csrc_dir().parent, _get_include_dir()],
    )
    logger.info("Generated Cake fused QK RoPE append JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fused_qk_rope_append_module(stage: str, arch: str):
    """Build and load one module; return ``(module, manifest record)``."""
    spec = gen_cake_fused_qk_rope_append_module(stage, arch)
    module = spec.build_and_load()
    return module, _record(stage, arch)


def build_all_cake_fused_qk_rope_append_modules(arch: str) -> dict[str, str]:
    built = {}
    for stage in STAGES:
        spec = gen_cake_fused_qk_rope_append_module(stage, arch)
        module = spec.build_and_load()
        record = _record(stage, arch)
        if not hasattr(module, str(record["ffi_entry"])):
            raise RuntimeError(f"built module {record['name']} lacks its FFI entry")
        built[stage] = spec.name
    return built
