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
"""JIT loader for the generated Cake fused QK RMSNorm + NeoX RoPE + FP8 quantize + paged KV append kernel.

One source-only manifest lives under ``csrc/cake_fused_qk_rope_fp8_append``
(``cake_fused_qk_rope_fp8_append_manifest.json``, schema ``cake.library_export.v5``):
a single stage (``fp8_e4m3``; head configuration, quantization / norm policies,
prefill vs decode and the caller-owned K/V route are launch parameters) whose one
record carries a ``builds`` entry (module name, kernel symbol) for each of
``sm_90a`` (H100/H200), ``sm_100a`` (B200) and ``sm_103a`` (B300), and
``contract.arg_plan`` with the binding's argument order.
The three modules compile the same kernel and tvm-ffi binding sources; the only
per-architecture inputs are the ``-gencode`` flags and the kernel symbol, which the
loader passes as ``-DCAKE_KERNEL_SYMBOL=<kernel_symbol>`` so that every module
keeps its own symbol when they are loaded into one process.  The host wrapper
lives in :mod:`flashinfer.cake_fused_qk_rope_fp8_append`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
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

_PACKAGE = "cake_fused_qk_rope_fp8_append"
_GENERATED_ROOT = f"csrc/{_PACKAGE}"
_MANIFEST_NAME = f"{_PACKAGE}_manifest.json"

STAGES = ("fp8_e4m3",)
ARCHES = ("sm_90a", "sm_100a", "sm_103a")
_NVCC_FLAGS = {
    "sm_90a": sm90a_nvcc_flags,
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
# The shared kernel and binding sources name the kernel through this macro; the
# loader binds it per module so that every module keeps a distinct symbol.
_KERNEL_SYMBOL_DEFINE = "CAKE_KERNEL_SYMBOL"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def arch_for(compute_capability: tuple[int, int]) -> str:
    major, minor = compute_capability
    if (major, minor) == (9, 0):
        return "sm_90a"
    if (major, minor) == (10, 0):
        return "sm_100a"
    if (major, minor) == (10, 3):
        return "sm_103a"
    raise ValueError(
        f"cake_fused_qk_rope_fp8_append has no cubin for compute capability {major}.{minor}; "
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


def arg_plan() -> tuple[tuple[str, str], ...]:
    """Binding arguments ``(kind, name)`` in call order (``contract.arg_plan``)."""
    plan = _manifest()["contract"]["arg_plan"]
    return tuple((str(kind), str(name)) for kind, name in plan)


def _record(stage: str, arch: str) -> dict[str, Any]:
    """Per-module view of a stage record: its shared fields plus ``builds[arch]``."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHES}")
    matches = [item for item in _manifest()["modules"] if item.get("stage") == stage]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Cake stage record for {stage!r}, got {len(matches)}"
        )
    record = matches[0]
    build = dict(record.get("builds", {})).get(arch)
    if not isinstance(build, dict):
        raise RuntimeError(f"stage {stage!r} ships no build for {arch}")
    view = {key: value for key, value in record.items() if key != "builds"}
    return {**view, **build, "arch": arch}


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


def _kernel_symbol_flag(record: dict[str, Any]) -> str:
    """``-D`` flag binding the shared sources to this module's kernel symbol."""
    symbol = record.get("kernel_symbol")
    if not isinstance(symbol, str) or _IDENTIFIER.fullmatch(symbol) is None:
        raise RuntimeError(
            f"invalid kernel_symbol for module {record.get('name')!r}: {symbol!r}"
        )
    return f"-D{_KERNEL_SYMBOL_DEFINE}={symbol}"


@functools.cache
def gen_cake_fused_qk_rope_fp8_append_module(stage: str, arch: str) -> JitSpec:
    record = _record(stage, arch)
    units = record["translation_units"]
    spec = gen_jit_spec(
        name=f"{record['name']}_{arch}",
        sources=[_source_path(units["device"]), _source_path(units["binding"])],
        extra_cuda_cflags=[
            *_NVCC_FLAGS[arch],
            *record["compile_flags"],
            _kernel_symbol_flag(record),
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[_get_csrc_dir().parent, _get_include_dir()],
    )
    logger.info("Generated Cake fused QK RoPE append JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_cake_fused_qk_rope_fp8_append_module(stage: str, arch: str):
    """Build and load one module; return ``(module, manifest record)``."""
    spec = gen_cake_fused_qk_rope_fp8_append_module(stage, arch)
    module = spec.build_and_load()
    return module, _record(stage, arch)
