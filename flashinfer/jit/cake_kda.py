"""Manifest-verified JIT loader for the exported Cake KDA portfolio."""

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

CakeKDATarget = Literal["sm100a", "sm103a"]
CakeKDAFamily = Literal[
    "bounded_bf16_evolution",
    "bounded_fp32_serving",
    "bounded_fp32_affine_prefix",
    "unbounded_bf16_serving",
    "unbounded_affine_prefix",
]
CakeKDARole = Literal["main", "prepare", "chain", "map", "scan", "correction"]

_MANIFEST = "cake_kda_prefill_portfolio_export_manifest.json"
_TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=3",
}
_SHAPE_SUITE_COUNTS = {
    "evolution_29": 29,
    "kimi_k3_serving_48": 48,
    "unbounded_serving_5": 5,
    "affine_predecessor_7": 7,
}
_REQUIRED_PROBLEM_SHAPES = {
    "evolution_29:h96_uniform_n32_holdout",
    "evolution_29:h96_uniform_n64",
    "evolution_29:h96_uniform_n128_holdout",
    "evolution_29:h96_uniform_n256",
    "evolution_29:h16_fixed_32768_holdout",
    "evolution_29:h4_fixed_65536_holdout",
    "evolution_29:h96_irregular_tail_varlen",
    "affine_predecessor_7:h4_t8192",
    "affine_predecessor_7:h4_t16384",
    "affine_predecessor_7:h8_t8192",
    "affine_predecessor_7:h8_t16384",
    "affine_predecessor_7:h16_t8192",
    "affine_predecessor_7:h16_t16384",
    "affine_predecessor_7:h32_t16384_direct",
}
_EVOLUTION_POLICIES = {
    "direct_m128_generic",
    "direct_m128_h96_commit_order",
    "persistent_m128_h64_lpt",
    "direct_vtile_m128_generic",
    "direct_vtile_m128_h64_gate_order",
    "persistent_vtile_m128_h96_six_task",
    "persistent_vtile_m128_h64",
    "direct_m64_independent_value_split",
}
_SERVING_COMMON_POLICIES = {
    "bt16_prepare",
    "bt16_chain_m64_s8",
    "bt16_chain_m64_s9",
    "direct_m128_h12_pair_packed_beta",
    "direct_m128_h12_scalar_early_pack",
    "direct_m128_n16_h12_scalar",
    "direct_m128_legacy_inverse",
    "direct_m128_register_inverse",
    "independent_dvsplit_m64",
    "persistent_m128_recurrence_pieces",
    "scalar_chunk_lpt_m128_h96",
    "small_bh_owner_helper_m128",
}
_SERVING_SM100_POLICIES = _SERVING_COMMON_POLICIES | {"persistent_m128_whole_chain"}
_SERVING_SM103_POLICIES = _SERVING_COMMON_POLICIES | {
    "direct_m128_prediction_first_tensor_decay",
    "source_vtile_m128_direct",
    "source_vtile_m128_persistent_six_task",
}
_ROLE_BY_POLICY: dict[str, CakeKDARole] = {
    "bt16_prepare": "prepare",
    "bt16_chain_m64_s8": "chain",
    "bt16_chain_m64_s9": "chain",
    "affine_split_map": "map",
    "affine_prefix_scan": "scan",
    "affine_split_correction": "correction",
}


def _expected_module_keys() -> set[tuple[str, str, str, str]]:
    expected: set[tuple[str, str, str, str]] = set()
    for arch in _TARGET_ARCH.values():
        expected.update(
            (arch, "bounded_bf16_evolution", policy, "main")
            for policy in _EVOLUTION_POLICIES
        )
        expected.add(
            (arch, "unbounded_bf16_serving", "direct_m128_unbounded_softplus", "main")
        )
        expected.update(
            (
                arch,
                "bounded_fp32_affine_prefix",
                policy,
                _ROLE_BY_POLICY.get(policy, "main"),
            )
            for policy in (
                "affine_split_main",
                "affine_split_map",
                "affine_split_correction",
            )
        )
        expected.update(
            (
                arch,
                "unbounded_affine_prefix",
                policy,
                _ROLE_BY_POLICY.get(policy, "main"),
            )
            for policy in (
                "affine_split_main",
                "affine_split_map",
                "affine_prefix_scan",
                "affine_split_correction",
            )
        )
    expected.update(
        ("sm_100a", "bounded_fp32_serving", policy, _ROLE_BY_POLICY.get(policy, "main"))
        for policy in _SERVING_SM100_POLICIES
    )
    expected.update(
        ("sm_103a", "bounded_fp32_serving", policy, _ROLE_BY_POLICY.get(policy, "main"))
        for policy in _SERVING_SM103_POLICIES
    )
    return expected


@dataclass(frozen=True)
class CakeKDAModuleSpec:
    """One exact-architecture source closure from the standard Cake export."""

    target: CakeKDATarget
    family: CakeKDAFamily
    policy: str
    role: CakeKDARole
    name: str
    module_ident: str
    closure_sha256: str
    compile_flags: tuple[str, ...]
    device_path: Path
    binding_path: Path
    use_pdl: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid Cake KDA portfolio export manifest: {message}")


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
def get_cake_kda_module_specs() -> tuple[CakeKDAModuleSpec, ...]:
    """Load and verify the complete 89-shape, 60-module source closure."""

    csrc_dir = get_kda_csrc_dir()
    payload: Any = json.loads((csrc_dir / _MANIFEST).read_text())
    _require(isinstance(payload, dict), "root must be an object")
    _require(payload.get("schema") == "cake.library_export.v1", "schema mismatch")
    _require(payload.get("producer") == "cake", "producer mismatch")
    _require(payload.get("library") == "flashinfer", "library mismatch")
    _require(payload.get("name") == "cake_kda_prefill_portfolio", "name mismatch")
    _require(payload.get("artifact_kind") == "source_only", "artifact kind mismatch")

    build = payload.get("build_contract")
    _require(isinstance(build, dict), "build contract missing")
    _require(
        build.get("translation_unit_model") == "separate_device_and_binding",
        "device and binding translation units must compile separately",
    )
    _require(build.get("binary_payloads") is False, "binary payloads forbidden")
    infrastructure = build.get("target_infrastructure")
    _require(isinstance(infrastructure, dict), "target infrastructure missing")
    _require(
        infrastructure.get("required_headers") == ["tvm_ffi_utils.h"],
        "target header contract mismatch",
    )

    contract = payload.get("contract")
    _require(isinstance(contract, dict), "contract missing")
    _require(contract.get("architectures") == ["sm_100a", "sm_103a"], "architectures")
    suites = contract.get("shape_suites")
    _require(isinstance(suites, dict), "shape suites missing")
    _require(set(suites) == set(_SHAPE_SUITE_COUNTS), "shape-suite set mismatch")
    for suite, count in _SHAPE_SUITE_COUNTS.items():
        labels = suites.get(suite)
        _require(
            isinstance(labels, list)
            and len(labels) == count
            and len(set(labels)) == count,
            f"{suite} must contain {count} unique rows",
        )
    denominator = contract.get("shape_denominator")
    _require(
        isinstance(denominator, list)
        and len(denominator) == 89
        and len(set(denominator)) == 89
        and contract.get("shape_count") == 89,
        "shape denominator must contain 89 unique rows",
    )
    _require(
        {label for labels in suites.values() for label in labels} == set(denominator),
        "shape suites must exactly partition the denominator",
    )
    _require(
        set(contract.get("required_problem_shapes", ())) == _REQUIRED_PROBLEM_SHAPES,
        "predecessor/problem-shape continuity mismatch",
    )
    _require(
        contract.get("timing_requirement") == "per_shape_source_export_interleaved",
        "interleaved timing requirement missing",
    )

    files = payload.get("files")
    _require(isinstance(files, list) and len(files) == 64, "file inventory mismatch")
    file_sha256: dict[str, str] = {}
    for index, item in enumerate(files):
        _require(isinstance(item, dict), f"files[{index}] must be an object")
        raw_path = item.get("path")
        digest = item.get("sha256")
        _require(isinstance(raw_path, str), f"files[{index}].path missing")
        _require(
            isinstance(digest, str) and len(digest) == 64, f"files[{index}].sha256"
        )
        path = _resolve_source(csrc_dir, raw_path, f"files[{index}].path")
        _require(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest,
            f"files[{index}] hash",
        )
        _require(raw_path not in file_sha256, f"duplicate file {raw_path}")
        file_sha256[raw_path] = digest

    modules = payload.get("modules")
    _require(
        isinstance(modules, list) and len(modules) == 60, "module inventory mismatch"
    )
    target_by_arch = {arch: target for target, arch in _TARGET_ARCH.items()}
    expected = _expected_module_keys()
    observed: set[tuple[str, str, str, str]] = set()
    specs: list[CakeKDAModuleSpec] = []
    for index, item in enumerate(modules):
        label = f"modules[{index}]"
        _require(isinstance(item, dict), f"{label} must be an object")
        arch = item.get("arch")
        route = item.get("route")
        role = item.get("role")
        _require(arch in target_by_arch, f"{label}.arch unsupported")
        _require(isinstance(route, dict), f"{label}.route missing")
        family = route.get("family")
        policy = route.get("policy")
        key = (arch, family, policy, role)
        _require(key in expected, f"{label} unsupported route {key}")
        _require(key not in observed, f"duplicate module {key}")
        observed.add(key)

        units = item.get("translation_units")
        _require(isinstance(units, dict), f"{label}.translation_units")
        _require(units.get("compile_separately") is True, f"{label}.compile_separately")
        device_raw = units.get("device")
        binding_raw = units.get("binding")
        device_path = _resolve_source(csrc_dir, device_raw, f"{label}.device")
        binding_path = _resolve_source(csrc_dir, binding_raw, f"{label}.binding")
        _require(
            device_raw in file_sha256 and binding_raw in file_sha256, f"{label}.files"
        )
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
        module_ident = item.get("module_ident")
        closure_sha256 = item.get("closure_sha256")
        compile_flags = item.get("compile_flags")
        launch = item.get("launch")
        _require(item.get("ffi_entry") == "run", f"{label}.ffi_entry")
        _require(item.get("tma_abi") == "grid_constant", f"{label}.tma_abi")
        _require(isinstance(item.get("name"), str), f"{label}.name")
        _require(isinstance(module_ident, str), f"{label}.module_ident")
        _require(
            isinstance(closure_sha256, str) and len(closure_sha256) == 64,
            f"{label}.closure_sha256",
        )
        _require(
            isinstance(compile_flags, list)
            and all(isinstance(flag, str) for flag in compile_flags),
            f"{label}.compile_flags",
        )
        _require(isinstance(launch, dict), f"{label}.launch")
        specs.append(
            CakeKDAModuleSpec(
                target=target_by_arch[arch],
                family=family,
                policy=policy,
                role=role,
                name=item["name"],
                module_ident=module_ident,
                closure_sha256=closure_sha256,
                compile_flags=tuple(compile_flags),
                device_path=device_path,
                binding_path=binding_path,
                use_pdl=launch.get("use_pdl") is True,
            )
        )

    _require(observed == expected, "target/route module set mismatch")
    specs.sort(key=lambda spec: (spec.target, spec.family, spec.policy, spec.role))
    return tuple(specs)


def cake_kda_is_available() -> bool:
    return len(get_cake_kda_module_specs()) == 60


def get_cake_kda_module_spec(
    target: CakeKDATarget,
    family: CakeKDAFamily,
    policy: str,
    role: CakeKDARole = "main",
) -> CakeKDAModuleSpec:
    for spec in get_cake_kda_module_specs():
        if (spec.target, spec.family, spec.policy, spec.role) == (
            target,
            family,
            policy,
            role,
        ):
            return spec
    raise ValueError(f"unsupported Cake KDA module: {target}/{family}/{policy}/{role}")


@functools.cache
def gen_cake_kda_module(
    target: CakeKDATarget,
    family: CakeKDAFamily,
    policy: str,
    role: CakeKDARole = "main",
) -> JitSpec:
    spec = get_cake_kda_module_spec(target, family, policy, role)
    jit_spec = gen_kda_jit_spec(
        name=f"{spec.module_ident}_{target}_{spec.closure_sha256}",
        sources=[spec.device_path, spec.binding_path],
        target=target,
        target_define=_TARGET_DEFINE[target],
        csrc_dir=get_kda_csrc_dir(),
        include_dir=get_flashinfer_include_dir(),
        extra_cuda_cflags=spec.compile_flags,
    )
    logger.info(
        "Generated Cake KDA portfolio JIT spec: "
        f"target={target}, family={family}, policy={policy}, role={role}"
    )
    return jit_spec


@functools.cache
def get_cake_kda_module(
    target: CakeKDATarget,
    family: CakeKDAFamily,
    policy: str,
    role: CakeKDARole = "main",
):
    return gen_cake_kda_module(target, family, policy, role).build_and_load()


__all__ = [
    "CakeKDAFamily",
    "CakeKDAModuleSpec",
    "CakeKDARole",
    "CakeKDATarget",
    "cake_kda_is_available",
    "gen_cake_kda_module",
    "get_cake_kda_module",
    "get_cake_kda_module_spec",
    "get_cake_kda_module_specs",
]
