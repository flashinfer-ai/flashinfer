"""JIT loader for the source-only SM103a concat MLA K backend."""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm103a_nvcc_flags

_MANIFEST_NAME = "cake_concat_mla_k_import_manifest.json"
_EXPECTED_CONTRACT = {
    "backend": "cake",
    "correctness": "byte_exact_copy_and_broadcast",
    "dtypes": [
        "bfloat16",
        "float16",
        "float8_e4m3fn",
        "float8_e5m2",
    ],
    "fixed_shape": {
        "nope_dim": 128,
        "num_heads": 128,
        "output_dim": 192,
        "rope_dim": 64,
    },
    "input_layouts": ["contiguous", "nope_strided", "both_strided"],
    "mutation": "caller_owned_k_in_place",
    "operator": "concat_mla_k",
    "output_layouts": [
        "caller_owned_uninitialized",
        "caller_owned_leading_strided_uninitialized",
    ],
    "public_api": "flashinfer.concat_ops.concat_mla_k",
    "return_value": None,
    "signature": "concat_mla_k(k, k_nope, k_rope) -> None",
    "target": "sm_103a",
}
_EXPECTED_BUILD_CONTRACT = {
    "architecture_source": "modules[].arch",
    "binary_payloads": False,
    "target_infrastructure": {
        "binding_runtime": "flashinfer_tvm_ffi_utils",
        "headers_owned_by_target": True,
        "required_headers": ["tvm_ffi_utils.h"],
    },
    "translation_unit_model": "separate_device_and_binding",
}


@dataclass(frozen=True)
class CakeConcatMLAKModuleSpec:
    """Verified source closure and cache identity for the SM103a module."""

    module_ident: str
    closure_sha256: str
    device_path: Path
    binding_path: Path


def _require_manifest(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid Cake concat MLA K import manifest: {message}")


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "concat_mla"
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "concat_mla"
    for candidate in (installed, checkout):
        if (candidate / _MANIFEST_NAME).is_file():
            return candidate
    raise FileNotFoundError(
        "Cake concat MLA K sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def _verify_source(
    csrc_dir: Path, path_value: object, sha256_value: object, label: str
) -> Path:
    _require_manifest(isinstance(path_value, str) and bool(path_value), f"{label}.path")
    assert isinstance(path_value, str)
    relative = PurePosixPath(path_value)
    _require_manifest(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("csrc", "concat_mla")
        and len(relative.parts) == 3,
        f"{label}.path must name one csrc/concat_mla file",
    )
    path = csrc_dir / relative.name
    _require_manifest(
        path.name.startswith("cake_concat_mla_k_") and path.suffix == ".cu",
        f"{label}.path must use a cake_concat_mla_k CUDA filename",
    )
    _require_manifest(path.is_file(), f"{label}.path does not exist: {path}")
    _require_manifest(
        isinstance(sha256_value, str)
        and len(sha256_value) == 64
        and all(character in "0123456789abcdef" for character in sha256_value),
        f"{label}.sha256 must be one full lowercase SHA-256",
    )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    _require_manifest(
        actual == sha256_value,
        f"{label}.sha256 mismatch: {actual} != {sha256_value}",
    )
    return path


def _compact_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@functools.cache
def get_cake_concat_mla_k_module_spec() -> CakeConcatMLAKModuleSpec:
    """Read and verify the complete source-only module receipt."""

    csrc_dir = _get_csrc_dir()
    payload: Any = json.loads((csrc_dir / _MANIFEST_NAME).read_text())
    _require_manifest(isinstance(payload, dict), "root must be an object")
    _require_manifest(payload.get("schema") == "cake.library_export.v1", "schema")
    _require_manifest(payload.get("producer") == "cake", "producer")
    _require_manifest(payload.get("artifact_kind") == "source_only", "artifact_kind")
    _require_manifest(payload.get("library") == "flashinfer", "library")
    _require_manifest(payload.get("name") == "cake_concat_mla_k", "name")
    _require_manifest(payload.get("contract") == _EXPECTED_CONTRACT, "contract")
    _require_manifest(
        payload.get("build_contract") == _EXPECTED_BUILD_CONTRACT,
        "build_contract",
    )

    modules = payload.get("modules")
    _require_manifest(isinstance(modules, list) and len(modules) == 1, "modules")
    module = modules[0]
    _require_manifest(isinstance(module, dict), "modules[0] must be an object")
    _require_manifest(module.get("arch") == "sm_103a", "modules[0].arch")
    _require_manifest(
        module.get("name") == "cake_concat_mla_k_vector_copy", "modules[0].name"
    )
    _require_manifest(module.get("role") == "main", "modules[0].role")
    _require_manifest(module.get("ffi_entry") == "run", "modules[0].ffi_entry")
    _require_manifest(
        module.get("compile_flags") == sm103a_nvcc_flags,
        "modules[0].compile_flags",
    )
    arg_plan = module.get("arg_plan")
    _require_manifest(
        isinstance(arg_plan, list) and bool(arg_plan), "modules[0].arg_plan"
    )
    _require_manifest(
        module.get("arg_plan_sha256") == _compact_sha256(arg_plan),
        "modules[0].arg_plan_sha256",
    )
    translation_units = module.get("translation_units")
    _require_manifest(
        isinstance(translation_units, dict)
        and translation_units.get("compile_separately") is True,
        "modules[0].translation_units",
    )

    closure = module.get("closure")
    _require_manifest(
        isinstance(closure, list) and len(closure) == 2,
        "modules[0].closure",
    )
    by_path = {
        item.get("path"): item
        for item in closure
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    device_value = translation_units.get("device")
    binding_value = translation_units.get("binding")
    _require_manifest(device_value in by_path, "device closure missing")
    _require_manifest(binding_value in by_path, "binding closure missing")
    device_item = by_path[device_value]
    binding_item = by_path[binding_value]
    device_path = _verify_source(
        csrc_dir,
        device_value,
        device_item.get("sha256"),
        "modules[0].device",
    )
    binding_path = _verify_source(
        csrc_dir,
        binding_value,
        binding_item.get("sha256"),
        "modules[0].binding",
    )
    files = payload.get("files")
    _require_manifest(isinstance(files, list) and len(files) == 2, "files")
    file_hashes = {
        item.get("path"): item.get("sha256")
        for item in files
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("sha256"), str)
    }
    _require_manifest(
        file_hashes
        == {
            device_value: device_item.get("sha256"),
            binding_value: binding_item.get("sha256"),
        },
        "files must match the complete module closure",
    )

    module_ident = module.get("module_ident")
    closure_sha256 = module.get("closure_sha256")
    _require_manifest(
        isinstance(module_ident, str)
        and module_ident.startswith("cake_concat_mla_k_")
        and module_ident.replace("_", "").isalnum(),
        "modules[0].module_ident",
    )
    _require_manifest(
        isinstance(closure_sha256, str)
        and len(closure_sha256) == 64
        and all(character in "0123456789abcdef" for character in closure_sha256),
        "modules[0].closure_sha256",
    )
    identity_input = {
        key: module[key]
        for key in (
            "arch",
            "name",
            "role",
            "translation_units",
            "kernel_symbol",
            "module_ident",
            "ffi_entry",
            "binding_infrastructure",
            "arg_plan",
            "compile_flags",
            "tma_abi",
            "launch",
            "route",
            "closure",
        )
    }
    _require_manifest(
        closure_sha256 == _compact_sha256(identity_input),
        "modules[0].closure_sha256 mismatch",
    )
    return CakeConcatMLAKModuleSpec(
        module_ident=module_ident,
        closure_sha256=closure_sha256,
        device_path=device_path,
        binding_path=binding_path,
    )


def get_cake_concat_mla_k_uri() -> str:
    spec = get_cake_concat_mla_k_module_spec()
    return f"{spec.module_ident}_sm103a_{spec.closure_sha256}"


@functools.cache
def gen_cake_concat_mla_k_module() -> JitSpec:
    """Generate the exact-SM103a JIT module from separate source units."""

    spec = get_cake_concat_mla_k_module_spec()
    csrc_dir = _get_csrc_dir()
    jit_spec = gen_jit_spec(
        name=get_cake_concat_mla_k_uri(),
        sources=[spec.device_path, spec.binding_path],
        extra_cuda_cflags=[*sm103a_nvcc_flags],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
        needs_device_linking=True,
    )
    logger.info("Generated Cake concat MLA K SM103a JIT spec: %s", jit_spec.name)
    return jit_spec


@functools.cache
def load_cake_concat_mla_k_module():
    module = gen_cake_concat_mla_k_module().build_and_load()
    logger.info("Loaded Cake concat MLA K SM103a module")
    return module


def get_cake_concat_mla_k_module():
    return load_cake_concat_mla_k_module()


__all__ = [
    "CakeConcatMLAKModuleSpec",
    "gen_cake_concat_mla_k_module",
    "get_cake_concat_mla_k_module",
    "get_cake_concat_mla_k_module_spec",
    "get_cake_concat_mla_k_uri",
    "load_cake_concat_mla_k_module",
]
