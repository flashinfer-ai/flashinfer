"""Manifest-verified JIT loader for the shared Cake KDA prefill export."""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from ._kda_jit_common import (
    gen_kda_jit_spec,
    get_flashinfer_include_dir,
    get_kda_csrc_dir,
)
from .core import JitSpec, logger

CakeKDAPrefillSharedTarget = Literal["sm100a", "sm103a"]
CakeKDAPrefillSharedPolicy = Literal[
    "direct_m128_generic",
    "direct_m128_h96_commit_order",
    "persistent_m128_h64_lpt",
    "direct_vtile_m128_generic",
    "direct_vtile_m128_h64_gate_order",
    "persistent_vtile_m128_h96_six_task",
    "persistent_vtile_m128_h64",
    "direct_m64_independent_value_split",
]

_MANIFEST = "cake_kda_prefill_shared_export_manifest.json"
_TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_CAKE_KDA_PREFILL_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_CAKE_KDA_PREFILL_TARGET_MINOR=3",
}
_EXPECTED_POLICIES: tuple[CakeKDAPrefillSharedPolicy, ...] = (
    "direct_m128_generic",
    "direct_m128_h96_commit_order",
    "persistent_m128_h64_lpt",
    "direct_vtile_m128_generic",
    "direct_vtile_m128_h64_gate_order",
    "persistent_vtile_m128_h96_six_task",
    "persistent_vtile_m128_h64",
    "direct_m64_independent_value_split",
)
_REQUIRED_PROBLEM_SHAPES = {
    "h96_uniform_n32_holdout",
    "h96_uniform_n64",
    "h96_uniform_n128_holdout",
    "h96_uniform_n256",
    "h16_fixed_32768_holdout",
    "h4_fixed_65536_holdout",
    "h96_irregular_tail_varlen",
}


@dataclass(frozen=True)
class CakeKDAPrefillSharedModuleSpec:
    """One target-specific receipt from the sealed source-only export."""

    target: CakeKDAPrefillSharedTarget
    policy: CakeKDAPrefillSharedPolicy
    name: str
    module_ident: str
    closure_sha256: str
    compile_flags: tuple[str, ...]
    device_path: Path
    binding_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid Cake KDA prefill export manifest: {message}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_source(csrc_dir: Path, raw_path: object, label: str) -> Path:
    _require(isinstance(raw_path, str), f"{label} must be a path")
    assert isinstance(raw_path, str)
    relative = PurePosixPath(raw_path)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("csrc", "kda")
        and len(relative.parts) == 3,
        f"{label} must name one csrc/kda file",
    )
    path = csrc_dir / relative.name
    _require(path.is_file(), f"{label} does not exist: {path}")
    return path


@functools.cache
def get_cake_kda_prefill_shared_module_specs() -> tuple[
    CakeKDAPrefillSharedModuleSpec, ...
]:
    """Load and verify the complete generated source closure."""

    csrc_dir = get_kda_csrc_dir()
    manifest_path = csrc_dir / _MANIFEST
    payload: Any = json.loads(manifest_path.read_text())
    _require(isinstance(payload, dict), "root must be an object")
    _require(payload.get("schema") == "cake.library_export.v1", "schema mismatch")
    _require(payload.get("producer") == "cake", "producer mismatch")
    _require(payload.get("library") == "flashinfer", "library mismatch")
    _require(payload.get("name") == "cake_kda_prefill_shared", "name mismatch")
    _require(payload.get("artifact_kind") == "source_only", "artifact kind mismatch")

    build_contract = payload.get("build_contract")
    _require(isinstance(build_contract, dict), "build_contract must be an object")
    _require(
        build_contract.get("translation_unit_model")
        == "separate_device_and_binding",
        "device and binding translation units must compile separately",
    )
    _require(build_contract.get("binary_payloads") is False, "binary payloads forbidden")
    infrastructure = build_contract.get("target_infrastructure")
    _require(isinstance(infrastructure, dict), "target infrastructure missing")
    _require(
        infrastructure.get("required_headers") == ["tvm_ffi_utils.h"],
        "target header contract mismatch",
    )

    contract = payload.get("contract")
    _require(isinstance(contract, dict), "contract must be an object")
    shapes = contract.get("shape_denominator")
    _require(
        isinstance(shapes, list)
        and len(shapes) == 29
        and len(set(shapes)) == 29
        and contract.get("shape_count") == 29,
        "shape denominator must contain 29 unique rows",
    )
    _require(
        _REQUIRED_PROBLEM_SHAPES.issubset(set(shapes)),
        "predecessor problem-shape continuity is incomplete",
    )

    files = payload.get("files")
    _require(isinstance(files, list) and len(files) == 16, "file inventory mismatch")
    file_sha256: dict[str, str] = {}
    for index, item in enumerate(files):
        _require(isinstance(item, dict), f"files[{index}] must be an object")
        raw_path = item.get("path")
        digest = item.get("sha256")
        _require(isinstance(raw_path, str), f"files[{index}].path missing")
        _require(
            isinstance(digest, str) and len(digest) == 64,
            f"files[{index}].sha256 invalid",
        )
        path = _resolve_source(csrc_dir, raw_path, f"files[{index}].path")
        _require(_sha256(path) == digest, f"files[{index}] SHA-256 mismatch")
        _require(raw_path not in file_sha256, f"duplicate file {raw_path}")
        file_sha256[raw_path] = digest

    modules = payload.get("modules")
    _require(isinstance(modules, list) and len(modules) == 16, "module inventory mismatch")
    expected = {
        (arch, policy)
        for arch in _TARGET_ARCH.values()
        for policy in _EXPECTED_POLICIES
    }
    observed: set[tuple[str, str]] = set()
    specs: list[CakeKDAPrefillSharedModuleSpec] = []
    target_by_arch = {arch: target for target, arch in _TARGET_ARCH.items()}
    for index, item in enumerate(modules):
        label = f"modules[{index}]"
        _require(isinstance(item, dict), f"{label} must be an object")
        arch = item.get("arch")
        route = item.get("route")
        _require(arch in target_by_arch, f"{label}.arch unsupported")
        _require(isinstance(route, dict), f"{label}.route missing")
        policy = route.get("policy")
        _require(policy in _EXPECTED_POLICIES, f"{label}.route.policy unsupported")
        key = (arch, policy)
        _require(key not in observed, f"duplicate module {key}")
        observed.add(key)

        translation_units = item.get("translation_units")
        _require(isinstance(translation_units, dict), f"{label}.translation_units missing")
        _require(
            translation_units.get("compile_separately") is True,
            f"{label} must compile translation units separately",
        )
        device_raw = translation_units.get("device")
        binding_raw = translation_units.get("binding")
        device_path = _resolve_source(csrc_dir, device_raw, f"{label}.device")
        binding_path = _resolve_source(csrc_dir, binding_raw, f"{label}.binding")
        _require(device_raw in file_sha256, f"{label}.device absent from file inventory")
        _require(binding_raw in file_sha256, f"{label}.binding absent from file inventory")

        closure = item.get("closure")
        _require(isinstance(closure, list) and len(closure) == 2, f"{label}.closure")
        closure_map = {
            entry.get("path"): entry.get("sha256")
            for entry in closure
            if isinstance(entry, dict)
        }
        _require(
            closure_map
            == {
                device_raw: file_sha256[device_raw],
                binding_raw: file_sha256[binding_raw],
            },
            f"{label}.closure mismatch",
        )
        _require(item.get("role") == "prefill", f"{label}.role mismatch")
        _require(item.get("ffi_entry") == "run", f"{label}.ffi_entry mismatch")
        _require(item.get("tma_abi") == "pointer", f"{label}.tma_abi mismatch")
        module_ident = item.get("module_ident")
        closure_sha256 = item.get("closure_sha256")
        compile_flags = item.get("compile_flags")
        name = item.get("name")
        _require(
            isinstance(name, str) and name.startswith("cake_kda_prefill_"),
            f"{label}.name invalid",
        )
        _require(
            isinstance(module_ident, str) and module_ident.startswith("cake_kda_prefill_"),
            f"{label}.module_ident invalid",
        )
        _require(
            isinstance(closure_sha256, str) and len(closure_sha256) == 64,
            f"{label}.closure_sha256 invalid",
        )
        _require(
            isinstance(compile_flags, list)
            and all(isinstance(flag, str) for flag in compile_flags),
            f"{label}.compile_flags invalid",
        )
        specs.append(
            CakeKDAPrefillSharedModuleSpec(
                target=target_by_arch[arch],
                policy=policy,
                name=name,
                module_ident=module_ident,
                closure_sha256=closure_sha256,
                compile_flags=tuple(compile_flags),
                device_path=device_path,
                binding_path=binding_path,
            )
        )

    _require(observed == expected, "target/policy module set mismatch")
    specs.sort(key=lambda spec: (spec.target, _EXPECTED_POLICIES.index(spec.policy)))
    return tuple(specs)


def get_cake_kda_prefill_shared_module_spec(
    target: CakeKDAPrefillSharedTarget,
    policy: CakeKDAPrefillSharedPolicy,
) -> CakeKDAPrefillSharedModuleSpec:
    for spec in get_cake_kda_prefill_shared_module_specs():
        if spec.target == target and spec.policy == policy:
            return spec
    raise ValueError(f"unsupported Cake KDA prefill module: {target}/{policy}")


@functools.cache
def gen_cake_kda_prefill_shared_module(
    target: CakeKDAPrefillSharedTarget,
    policy: CakeKDAPrefillSharedPolicy,
) -> JitSpec:
    """Construct the JIT spec for one exact target and physical policy."""

    spec = get_cake_kda_prefill_shared_module_spec(target, policy)
    csrc_dir = get_kda_csrc_dir()
    jit_spec = gen_kda_jit_spec(
        name=f"{spec.module_ident}_{target}_{spec.closure_sha256}",
        sources=[spec.device_path, spec.binding_path],
        target=target,
        target_define=_TARGET_DEFINE[target],
        csrc_dir=csrc_dir,
        include_dir=get_flashinfer_include_dir(),
        extra_cuda_cflags=spec.compile_flags,
    )
    logger.info(
        "Generated shared Cake KDA prefill JIT spec: "
        f"target={target}, policy={policy}, name={jit_spec.name}"
    )
    return jit_spec


@functools.cache
def get_cake_kda_prefill_shared_module(
    target: CakeKDAPrefillSharedTarget,
    policy: CakeKDAPrefillSharedPolicy,
):
    """Build or load one manifest-verified generated module."""

    return gen_cake_kda_prefill_shared_module(target, policy).build_and_load()


__all__ = [
    "CakeKDAPrefillSharedModuleSpec",
    "CakeKDAPrefillSharedPolicy",
    "CakeKDAPrefillSharedTarget",
    "gen_cake_kda_prefill_shared_module",
    "get_cake_kda_prefill_shared_module",
    "get_cake_kda_prefill_shared_module_spec",
    "get_cake_kda_prefill_shared_module_specs",
]
